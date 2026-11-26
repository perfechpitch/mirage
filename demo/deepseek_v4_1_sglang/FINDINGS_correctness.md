# DeepSeek-V4.1-Flash 静态 megakernel —— 正确性 Findings

日期：2026-11-26
目标：`DSV41_STATIC_ATTN=all / DSV41_STATIC_MOE=all / DSV41_STATIC_POOL=1` 的 40 层 decode
与 stock SGLang 的 next-token 分布一致（对 `golden_bench32k.jsonl`），无 NaN、不塌到 token 0。

姊妹文档：`FINDINGS_bs1.md`（bs=1 延迟那轮）。本文只写**正确性**这条线，重点是**证据时效**和**踩过的坑**。

---

## 1. 结论速览

| 项 | 状态 | 证据 |
|---|---|---|
| 40 层单进程链（demo 随机 cache，mHC 开） | ✅ 跑通，**0 NaN** | `full40.log`：40×2 step 全部 `nf 0`，一个 step 80 次 launch 合计 4030 µs |
| 40 层单进程链（**真实 32K prefill KV**） | ❌ **从未跑完** | `fullreal.log`：构建完 80 个后跑到 a10/m10 被 **SIGKILL**，不是代码失败 |
| serving 路径（多进程 + mHC） | 🔧 **根因已定位并修好（待 GPU 验证）** | 唯一没有跨 rank 栅栏的路径正是唯一 NaN 的路径（见 §4） |
| prefill KV cache 里有 NaN 字节 | ❌ 已排除 | `compress_cache` e4m3 0 个；`window`/`index_k` ue8m0 0 个 `0xFF` |
| L4 权重损坏 | ❌ 已排除 | 24 个张量无 nonfinite、无全零 |
| reuse 层数学（共享 L2 的 cache/idx） | ✅ 正确 | `l2_pool3.log`：L0–L6 relRMS 3.6e-2…6.8e-2（噪声量级） |
| 与 golden 对齐 | ❌ 未验证 | 真实 KV 那次没跑完 |

**一句话**：不是"算错了"，是**服务态纯静态链少了跨 rank 栅栏**（§4），加上另外两个工程故障没清掉——
① 证据早于代码修复（§2），② 长跑被 SIGKILL（§7.1）。栅栏已修并在 CPU 上验证编译，待 GPU 上做一正一反验证。

---

## 2. 证据时效 —— 最重要的一条教训

`goal` 里写的是"attn L4 (reuse) 在输入有限时输出 NaN"。核对原始日志后，这句话**只对了一半**：

| 日志 | 时间 | 配置 | 首个 NaN |
|---|---|---|---|
| `e2e_t1.log` | 11-25 13:50 | `e2e_decode.py`（多进程，mHC） | **attn L3** |
| `e2e40_dbg.log` | 11-25 08:42 | 同上 | **attn L3** |
| `l2_nan.log` | 11-25 13:20 | `MHC=False, check=False, KEEP=static, POOL=True` | **attn L4**（x 有限、y NaN） |
| `l2_pool3.log` | 11-25 11:02 | `check=True`（每层先跑 SGLang 的 kernel） | 无 NaN，L0–L6 relRMS 正常 |
| `full40.log` | 11-26 01:57 | 单进程 `full_decode.py`，mHC，demo cache | 无 NaN |
| `fullreal.log` | 11-26 03:27 | 单进程 + 真实 KV | **被 SIGKILL** |

**同一个 bug，首个 NaN 的层在 L3 和 L4 之间移动** —— 这不是确定性数据错误的特征，是竞态的特征。

而 `dsv41_static_hook.py` 的 mtime 是 **11-26 02:00**，加了 `TPBAR` + 每 sublayer 一次 barrier
（`dsv41_static_hook.py:193`，调用点 551 / 576 / 727）。**上面所有 NaN 日志都早于这个修复。**

> 教训 2.1：拿日志当证据前，先比 `ls -l` 的 mtime 和代码的 mtime。修复前的日志不能证明修复后还坏。
> 教训 2.2：goal 里复述现象时，别凭记忆写层号；`e2e_decode` 的 NaN 在 L3，`l2_check` 的在 L4，是两个配置。

---

## 3. 已排除项（附复现命令，全部 CPU 可做）

### 3.1 prefill cache 里没有 NaN 编码

dump 产物：`/tmp/prefill_row0`（968 个文件，8.4 GB，rank0 的 `r00_L{00..39}_{window_cache,compress_cache,index_k,cmp_state}.bin`）。

| 张量 | scale 格式 | NaN 编码 | 实测 |
|---|---|---|---|
| `window_cache` 528B/行（512 fp8 + 16 scale） | **ue8m0** | 只有 `0xFF`（=2^128=inf） | 0 个 |
| `index_k` 页 = [128 行×64B 值][128 行×4B scale] | **ue8m0** | 只有 `0xFF` | 0 个 |
| `compress_cache` 288B/行（256 e2m1 + 32 scale） | **e4m3** | `0x7F` 与 `0xFF` 都是 NaN | 0 个 |

> 坑 3.1：**三种 scale 不是同一格式**。把 ue8m0 当 e4m3 扫，会得到"L2 有 71/76800 个 NaN scale"这种假阳性
> （`0x7F` 在 ue8m0 里是 1.0，完全合法）。
> 坑 3.2：`index_k` **不是**每行 68 字节交错，而是页内先 128 行的 64 字节、再 128 行的 4 字节 scale。
> 按 `reshape(-1, 68)` 切会扫到完全错的位置（我第一次就错在这里，报了 69 行"NaN"）。

### 3.2 L4 的权重

`layers.4.*` 共 24 个张量（不含 routed experts / mtp），无 nonfinite、无全零。
`attn_sink` 的键名是 **`layers.4.attn.attn_sink`**（多一层 `attn.`），不是 `layers.4.attn_sink`。

### 3.3 reuse 层的共享路径

| 检查 | 结果 |
|---|---|
| L3 / L4 的 `compress_idx` 声明 | 完全一致：`tensor compress_idx 4096 -1 0` |
| indexer 与 sparse_attn 的 CAP | 一致：`attn2` 两者都是 **17408** |
| `compress_idx` 是否被 reset | **不会**：`tensor` 类型只绑指针（`include/mirage/static_megakernel/host.cuh:388`），host 和 kernel 都不 memset |
| reuse 的数学 | `l2_pool3.log`（真实 pool）L3 relRMS 6.3e-2、L4 6.0e-2 |

---

## 4. 根因：服务态纯静态链没有跨 rank 栅栏（唯一一条没栅栏的路径）

### 4.0 代码自己写明了这个缺口

`include/mirage/static_megakernel/core.cuh` 的 `kernel_begin` 有三条互斥分支：

| 行 | 分支 | 谁用 |
|---|---|---|
| 548 | `#ifdef STATIC_NO_START_BARRIER` | `DSV41_X_FLAG=1`（实验，默认关） |
| **584** | `#elif defined(STATIC_RESET_IN_KERNEL)` | **默认 serving 路径** |
| 690 | `#else` | host reset 路径（`full_decode.py`、`l2_check` 默认） |

而 584 那条分支的注释原文（`core.cuh:588-592`）：

> Every SM waits until every SM **of this GPU** has done its resets (...) **No barrier over the GPUs**:
> the exchange region has two sets (`EXCHANGE_SET`), and another GPU's launch n writes only into set
> n % 2, which this GPU last read in launch **n − 2**.

这段话把不变量说得很清楚，也把缺口说得很清楚：

- 跨 GPU **没有**栅栏，只有"本 GPU 内所有 SM 到齐"；
- 依赖的不变量是 **rank 间漂移 ≤ 1 次 launch**（这样 launch n 写的 set n%2，对端最早也要到 n−2 才读过）。

**漂移 2 次就正好击穿这个不变量**：落后 2 次的对端此刻正在读 launch n−2 用的
`(n−2) % 2 == n % 2` —— 正是领先方刚写的那一套。而每次 launch 的 reset 填充字节是 **`0xFF`**，
float 位模式就是 **NaN**。所以症状不是"数值不准"，是确定的 NaN。

### 4.1 为什么漂移会到 2 次

serving 路径下每个 sublayer 是 8 个进程各自 stream 上一次独立 launch，**层间没有任何 host 同步**。
8 个进程互相独立推进，落后 2 次是常态而不是意外。

### 4.2 三条路径的实测对照

| 路径 | reset 由谁做 | 层间 TP 对齐 | 结果 |
|---|---|---|---|
| `full_decode.py`（demo build） | host，每次 launch memset | ✅ 走 690 分支，自带 `push16(EXCHANGE_HELLO...)` 起点栅栏 | 干净（`full40.log`，80 次 launch，`nf 0`） |
| `l2_check` `CHECK=1`（每层先跑 SGLang kernel） | serving | ✅ SGLang 自己的 collective 顺便对齐 | 干净（`l2_pool3.log`，L0–L6 relRMS 正常） |
| **serving 纯静态链** | serving | ❌ 584 分支，无跨 GPU 栅栏 | **首个 reuse 层 NaN**（`l2_nan.log` L4 / `e2e_t1.log` L3） |

`full_decode.py:67` 明确注释 `FLAGS` 里不含 `STATIC_RESET_IN_KERNEL`（`the host reset path`），
与上表一致。**唯一没有栅栏的路径，正是唯一出 NaN 的路径。**

### 4.3 0xFF 与 NaN 到底在哪 —— 只有交换区是跨卡的

**先纠正一个我一开始写错的结论。** 我最初以为"每次 launch reset 的缓冲区"被抹成 `0xFF`、
而 peer 会去读它们，所以晚到的 rank 会毁掉 peer 的输入。**这是错的**：`g.cnt`（计数器）和
reset 列表都是 `dmalloc` 出来的**卡内私有**内存（`host.cuh:435`、`host.cuh:527`），
`core.cuh:588` 的原话就是 "the resets left to the start (the host's list: **buffers only this GPU
writes**)"。所以 reset 在任何位置都不会影响别的 rank。

真正跨卡的只有 **exchange 区**（`G::mc` / `G::rv`，"the exchange region: multicast address
(GPUS > 1), this GPU's copy (**2 sets**)"）。而 `MHC=1`（11-25 那次和这次复现都用它）的 build 里
它是有内容的 —— 实测 attn3_t1 / moe3_t1 的生成代码：

```
STATIC_EXCHANGE_SET     86016
STATIC_EXCHANGE_HELLO   81920
STATIC_EXCHANGE_OFFSETS 全 0
STATIC_EXCHANGE_BYTES   slot 23 = 81920      ← mHC sum_store 的 streams 交换缓冲
```

即：**每个 set 里 offset 0 有 81920 字节的跨卡交换缓冲**（mHC 的 streams all-reduce），
hello 槽在其后（81920 + rank*16，8 rank 共 128 字节），set 步长 86016。
（非 mHC 的 build 里 `BYTES` 全 0，确实没有交换缓冲 —— 这也是为什么 11-25 那次是 `MHC=1` 才炸。）

`0xFF` 是**交换区未写入时的初始填充**（就是本代码库的"尚未写入"哨兵，`runtime.cuh` 的
`valid16` / `validf4` 都在判它），float 位模式即 NaN。所以读到它就是读到 NaN，
与 `l2_nan.log` 的 `x finite=True / y NaN` 吻合。

### 4.4 为什么首个 NaN 落在第一个 reuse 层

两步：

1. **同一个 launch 内消费**：launch n 里，每个 rank 把 streams 写进 set `n % 2`，peer 在**同一个
   launch n** 从 set `n % 2` 读。所以"所有 rank 都在 launch n"是数据存在的**前提**。
2. **漂移 1 次就已经是错的**：落后一次的 peer 写的是 set `(n−1)%2`，当前 launch 要读的 set
   `n%2` 里它那一格还是 launch `n−2` 的旧值（若从没写过，就是 0xFF → NaN）。

而 reuse 层（L3/L4）是**第一个消费另一层产物**的层（读 L2 的 `compress_cache` + `compress_idx`），
L0–L2 只写自己的缓冲。所以竞态第一次有了"跨层传值"这个兑现点 —— 首次 NaN 在 L3/L4 之间移动，
取决于启动时序，是竞态而非确定性算错。

### 4.5 修法（已落到代码）

内核起点栅栏 `-DSTATIC_TP_BARRIER`，位置在 `core.cuh` 的 584 分支内、
**在所有 task 之前**（`kernel_begin` 开头，第 616 行起）。每个 SM 的 thread 0 等所有 rank 都发布了
本次 launch 的编号，然后 `__syncthreads()` 放行本 SM 的其余线程。

为什么用 `>= gen` 而不是 `== gen`，以及为什么它足够：
- **`>=` 是必须的**：领先一次的 peer 值更大。而它既然能到 n+1，就**必然已经跑完 n**，
  所以它 set `n%2` 的数据是齐的 —— 等它是安全的。若要求相等，一发生漂移就直接**死锁**。
- **`>=` 也足够**：一个 rank 不可能在本 rank 还停在 n 时越过 n+1 —— 它 n+1 的栅栏要求
  本 rank 已到 n+1。所以漂移被钳在**一次 launch**，恰好是两套 exchange 需要的不变量。
- "进入 launch n 就证明跑完了 n−1" 来自 **stream 串行**（PDL 已排除，见下），不依赖内存序。

> **一处我写错并已纠正**：本文件早先版本说"栅栏必须放在 reset 之前，否则 reset 会抹掉 peer 要读的
> 数据、导致死锁"。**不对** —— reset 列表是卡内私有内存（§4.3），放在哪都不影响 peer。
> 现在放在最前面是为了"任何 task 之前就先确认 peer 到齐"，语义干净；不是因为它不能放后面。
> 记录这个纠正，是因为**错误的原因比错误的代码更危险**：照错误原因去改，下次就会改坏。

`multimem.st` 会写包括本卡在内的所有副本，因此不会在 `r == g.rank` 上自旋死锁。

`multimem.st` 会写包括本卡在内的所有副本，因此不会在 `r == g.rank` 上自旋死锁。

**`multimem.st.weak` 没有 release 语义，这里不需要**（这条会被反复问到，记清楚）：
- 栅栏要保证的是"peer 已进入 launch n"，而"peer 的上一次 launch 的写入已可见"来自
  **stream 串行 + kernel 边界隐含的 memory fence**（PDL 已排除，见下）。在 launch n 里加
  `__threadfence()` 什么也换不来，因为那些写属于**上一个 kernel**，早已越过边界。
- "漂移 ≤ 1 次 launch" 是**纯计数**论证，不涉及内存序。
- 单个 32-bit 字不会撕裂；槽位单调递增，读到 `>= gen` 就说明对方至少到了 gen。
- host 路径（已由 `full40.log` 80 次 launch / 0 NaN 证明）用的是同一个 `multimem.st`。

**地址安全性**（也验过）：`push16_mc` 写 `EXCHANGE_HELLO + rank*16`，不加 `rg_set()`。规划器
`exchange_layout()` 为 hello 预留 `num_gpus*16` = 128 字节，且所有 buffer 的 offset 都 ≥
`align(hello + 128)`，所以 128 字节的头不会被任何 buffer 覆盖（实测 attn3/moe4/head 的
`STATIC_EXCHANGE_HELLO` 都是 0，最小的 buffer offset 是 4096）。`g.gen_ptr` 则是独立的
`dmalloc`（`host.cuh:527`），不在 exchange 区，也不会和 offset 0 撞。

- 之前它是**死代码**：`grep -rn STATIC_TP_BARRIER` 全仓只有 `#ifdef` 那处，没有任何 build 定义它。
- 现在 `StaticAttn` / `StaticMoE` / `StaticHead` 在 **serving（`-DSTATIC_RESET_IN_KERNEL`）且 `tp > 1`**
  时默认加上（`DSV41_STATIC_TPB=0` 关掉做计时）。host reset 路径不需要它（那条分支自带栅栏）。
- banner 打印 `TPBAR=` 与 `TPB=`；两者都为 0 且 `check=0` 且 8 rank 时打 WARNING。
- 与 `-DSTATIC_NO_START_BARRIER` 同时打开会**断言失败**（见下），不会静默失效。

#### 编译验证（CPU 上的 nvcc，不需要 GPU）

四组 flag 全部编译通过（attn3 / moe4 / head 三类真实生成的层）：

| 组合 | 结果 |
|---|---|
| `-DSTATIC_RESET_IN_KERNEL`（serving，无栅栏） | ✅ `.nv_fatbin` = 0x02de80 |
| `-DSTATIC_RESET_IN_KERNEL -DSTATIC_TP_BARRIER` | ✅ `.nv_fatbin` = 0x02df20（**+160 字节**设备代码） |
| 两者都不给（host 路径） | ✅ |
| `-DSTATIC_NO_START_BARRIER`（± `TP_BARRIER`） | ✅ 两种 fatbin 尺寸**完全相同** |

两个要点：

1. **`+160` 字节证明栅栏真的进了设备代码**。最初我以为"`.so` 尺寸相同 = 没编进去"，其实是段对齐把
   160 字节吸收了；看 `.nv_fatbin` 而非文件大小才看得出。另外 `nvcc -E` 里
   `EXCHANGE_HELLO + (size_t)g.rank * 16` 的出现次数是 1 次（带 flag）/ 0 次（不带），也是直接证据。
2. **`NO_START_BARRIER` ± `TP_BARRIER` 的 fatbin 尺寸完全相同**，直接证明了下面那个缺口。

**已知剩余缺口**：`STATIC_TP_BARRIER` 那块代码在 `#elif defined(STATIC_RESET_IN_KERNEL)` 分支内
（`core.cuh:584`），所以 **`STATIC_NO_START_BARRIER` 分支（548–583，`DSV41_X_FLAG=1` 的实验路径）
仍然没有跨 GPU 栅栏**。那条路径默认关闭；现在由 `dsv41_static_layer.py` 里的断言兜住 ——
两者同时出现时直接报错，而不是静默丢掉漂移界。将来若要启用该分支，得先给它也加一块。

#### build cache 会正确失效（否则验收会用到没栅栏的旧 so）

`static_schedule.py` 的缓存键是 `md5(layer.cu) + nvcc 前 4 个参数 + extra_flags + code_tag()`，
其中 `code_tag()`（`static_schedule.py:166`）是**所有** `include/mirage/static_megakernel/*.cuh`
和 `tasks/*.cuh` 的 md5。实测：

| commit | code_tag |
|---|---|
| `2a69b646`（栅栏改到 reset 之前之前） | `fff2dbcd6457` |
| `30ef1805`（当前） | `a6b3a5c7587c` |

**变了** —— 所以两条失效路径都成立：
1. `-DSTATIC_TP_BARRIER` 进 `extra_flags` → 键变；
2. 改 `core.cuh` 本身 → `code_tag` 变。

代价要说清楚：**现有 80 个 `.so`（含 `/tmp/full_real` 那套）已全部失效**，
下一次 serving 验收必然是一整次重编。**不要按"1.4 min × 80"估**：实测 6 层（12 个 layer-instance ×
8 rank = **96 次 nvcc**）只用了 **1.8 分钟**（8 个进程并行）。所以 40 层（80 × 8 = 640 次）
约 **12–15 分钟**，不是两小时。这不是意外的小开销，但也不是排期瓶颈。
好处是：不会出现"跑完了才发现用的是旧二进制"这种白烧一轮的情况。

#### PDL 与栅栏互斥

栅栏的语义依赖一个前提：**一个 rank 启动了 launch n，就说明它已经跑完 launch n−1**。
PDL（programmatic dependent launch, `-DSTATIC_PDL_TRIGGER=1`）恰好违反它 —— kernel 在上一个 kernel
还在跑时就启动，而 reset 就发生在 kernel 起点。所以两者不能同时开：

| `DSV41_STATIC_PDL` | `DSV41_STATIC_TPB` | 结果 |
|---|---|---|
| 0（默认） | 1（默认） | PDL=0, TPB=1 ✅ |
| 1 | 1 | **PDL 被强制关掉**，打 NOTE |
| 1 + `DSV41_STATIC_PDL_FORCE=1` | 1 | **TPB 被强制关掉**（写回环境变量，build 与 hook 一致） |
| 1 | 0 | PDL=1, TPB=0（纯计时跑） |

五种组合都实跑过决策表，没有"两个都开"的格子。PDL 默认是 0，所以默认 serving 路径本来就满足前提。

**为什么不再用 host 侧 `TPBAR`（NCCL）**：它每个 sublayer 一次，一个 step 80 次，bs=1 延迟敏感；
内核栅栏没有 host 往返。两者都在，`TPBAR` 仍按 `MHC` 决定，保留作为对照。

`l2_nan.log` 的另外两个细节支持它：`check=False, KEEP=static, T=1, POOL=True`；
L0–L3 全部有限，L4 的 `x finite=True` 但 `y NaN` —— 输入没问题，是 L4 kernel 自己**读**到了哨兵值。

---

## 5. 两个已修的数学 bug（真 bug，不是环境）

1. **mHC 初始残差必须 `hc_broadcast`**。SGLang 的残差流起点是
   `hidden_states = hc_broadcast(hidden_states, hc_mult=4)`，即 embedding 复制进 4 条流。
   `full_decode.py` 早期传的是全 0，导致 L0 的 `hc_pre` 混合了 4 条空流，整条 40 层链从第一步就偏。
   现在：`S0[r][0].copy_(e.repeat(HC))`。
2. **head logits 不能按 vocab 切分**。`gather_f32_layer` 已经把完整 `[T, 129280]` 收进**每个** rank 的
   `logits`，再按 `V//8` 切片拼接会把前 16160 个 vocab 重复 8 次。现在只读 rank 0 的完整分布。

（这两条都是 `full_decode.py` 里已经改好的；记在这里是因为它们是"算错"而不是"环境错"，容易再犯。）

---

## 6. 参考本身有噪声底 —— 验收标准必须据此定

`golden_bench32k.jsonl` 第 0 行（row 0）实测：

```
prompt_tokens 31906, tp 8, deterministic False
ids[:4]        [4588, 343, 6333, 294]
top1           (4588,  -1.7598)
top2           (8060,  -1.8848)      gap 0.125
```

> **`deterministic: False`**。而且 `bisect_repeat.sh` 的头注释已经记了实测：同一个 row、同一配置，
> 两个全新 engine，前 5 个 id 相同、**第 6 个开始分叉**，logprob 相差 **~0.1**。

所以：**跑 32 步逐 token 全等是不可达的**，别拿它当门槛。合理门槛（也是 goal 的口径）：

- 无 NaN、不塌到 token 0；
- step 0 的 **top-32 集合重叠** 与 **logprob 差 ≤ 1e-3**（第 0 步还没有累积漂移，是最干净的判据）；
- 后续步只要求 top-1 落在参考 top-5 内，且 logprob 差 ≤ 噪声底（~0.1）。

---

## 7. 不要重蹈覆辙 —— 操作清单

### 7.1 进程

1. **不要用 `pkill -f <宽模式>`**。曾经 `pkill -f "sglang::scheduler"` 误杀了同账号下用户自己的
   `dsv41_pdl_e2e_tp8.py bench`（PID 2584628），被明确批评过。
   正确做法：`nvidia-smi --query-compute-apps=pid,used_memory` 取 PID，**只杀自己启动的**，
   循环里先打印再杀。
2. **这台机的 `Killed` 先查清楚是 OOM 还是被杀**。现象：`/tmp` 下 21 个日志都带 `Killed`。
   `fullreal.log` 被 Killed 时 GPU 还有 219 GB 空闲、host 有 3.1 TB available → **不是显存 OOM**。
   共享机上 `user.slice` 的内存里大头是 page cache（权重/dump），并发大作业时会被挤，
   先 `free -g` + `nvidia-smi` 再决定跑不跑。
3. **跑之前看两张表**：`nvidia-smi --query-gpu=index,memory.used`（有没有别人的作业）和 `df -h /dev/shm`（§8.7 of bs1）。

### 7.2 构建

4. **`--out` 不要每次换新目录**。`full_decode.py --out DIR` 决定 build 目录；
   换目录 = 80 个 sublayer 全部用 nvcc 重编译（`fullreal.sh` 曾指到全新的 `/tmp/full_real`）。
   现有可用：`/tmp/full_L40`（80 个，195 MB）与 `/tmp/full_real`（80 个，已建好）。
   **直接复用 `/tmp/full_real`**，构建阶段应当全部命中缓存（日志会出现 `the layer from the build cache`）。

### 7.3 SGLang 侧

5. `--mem-fraction-static` 低于 **0.669** 会在 KV 分配阶段直接报错（`sglcheck.log`：
   `Raise --mem-fraction-static above 0.669`）。`dump_prefill.py` 早期用 0.4，白跑一轮。
6. `SGLANG_DSV4_KV_LAYOUT=v41_new` 必须设，否则 `KVLayout.V4` assert 直接挂（bs1 §8.4 已记）。
7. 别把**性能 flag**和**正确性验证**混在一次跑里（bs1 §8.6 的教训：`STATIC_NO_START_BARRIER`、poll ns=0 那一套）。

### 7.4 代码

8. `git checkout --ours <file>` 会连**非冲突**的对侧改动一起丢掉（bs1 §4）。冲突文件要手工取并集。

---

## 8. 下一步（命令可直接复制）

### 8.1 证实"缺栅栏"就是根因

> 判读入口：banner 打印 `TPBAR=`、`TPB=`、`PDL=`。看到 `check=False`、`TPBAR=0`、`TPB=0` 且 `PDL=0`
> 的日志，就**不能**用它论证"kernel 算错了"（那种配置本来就缺跨 rank 顺序）。

**一条命令跑完一正一反**（`verify_tpbar.sh`，三种配置互相对照）：

```bash
cd /opt/dlami/nvme/saboxu/mirage/demo/deepseek_v4_1_sglang
./verify_tpbar.sh                      # 6 层：只建 12 个 kernel，够复现 L3/L4
DSV41_E2E_LAYERS=40 ./verify_tpbar.sh  # 完整 40 层（80 个 kernel）
```

| tag | 配置 | 期望 |
|---|---|---|
| `tpb1` | `TPB=1`（默认） | 全部 `x_out` finite ✅ |
| `tpb0_hostsync` | `TPB=0` + 每层 `dist.barrier()` | finite（安全基线，但每 sublayer 付一次 host 往返） |
| `tpb0_nosync` | `TPB=0` + `DSV41_E2E_NOSYNC=1` | **NOT FINITE**，首个在 L3 或 L4 ← 反向对照 |

`tpb0_nosync` 就是 11-25 出 NaN 的那个配置。前两个共用一套 `TPB=0` build，所以只建两套。

#### 已跑到的结果（11-26）

反向对照 **已证实**。`DSV41_E2E_LAYERS=6`、`TPB=0`、`DSV41_E2E_NOSYNC=1`（= 11-25 配置）：

```
[e2e] built 6 layers in 1.8 min on rank 0
[e2e] NOT FINITE in 6 sublayers, the first is attn L3: attn L3, moe L3, attn L4, moe L4, attn L5, moe L5
```

- 首个非有限就是 **`attn L3`**，之后 L3–L5 全毁 —— 与 11-25 `e2e_t1.log` 的 L3 NaN **一字不差**。
- 而这一步的输入是有限的（同一份随机 KV，11-26 的白盒扫描 0 个非有限 scale 字节），
  所以不是"输入带 NaN"，是 kernel 自己**读**到了 `0xFF` 哨兵。
- 之后进程在 warmup 那一步**挂住**（`dist.barrier()` 再也到不了，有 rank 在 kernel 里自旋）。
  所以这条对照必须"只跑一步"，否则会白等：这正是 `DSV41_E2E_ONE=1` 存在的原因，
  也是"漂移"的第二个表现（不只是 NaN，还会死锁）。

正向（`TPB=1`）与 `tpb0_hostsync` 两档待 GPU 空闲后补跑。

**为什么不用 `l2_check.py` 做这条对照**：`CHECK=0` 的纯静态链会让它在没比过任何 sublayer 时
主动 `exit FAILED`（"a check that compared nothing must not read as a pass"）—— 那是它该做的；
而 `CHECK=1` 每层先跑 SGLang 的 kernel，那些 collective 顺便把 rank 对齐了，**掩盖**了漂移。
所以 NaN 的正反对照放在 `e2e_decode.py`，数值验收放在 8.3。

**`e2e_decode.py` 本身改了两处**（都是我 11-26 那次改动带来的副作用，已修正）：

1. 每层后的 `_sync()` 在 `TPB=1` 时**直接返回**。原来是无条件 `dist.barrier()`：
   既掩盖了漂移，又让这个计时工具每 sublayer 多付一次 host 往返，**测出来的 TPOT 不再是纯静态链的**。
2. 步末新增逐层 finite 判定，打印 `all 40 attn + 40 moe x_out finite` 或
   `NOT FINITE in N sublayers, the first is attn L3: ...`。正常路径也打印，不再只在 DEBUG 下才看得到。

`_sync()` 的真值表（四种组合都实跑过）：

| `DSV41_STATIC_TPB` | `DSV41_E2E_NOSYNC` | kernel 栅栏 | host barrier |
|---|---|---|---|
| 1（默认） | 0 | ON | 跳过 |
| 0 | 0 | OFF | 调用 |
| 0 | 1 | OFF | 跳过 ← 复现 |
| 1 | 1 | ON | 跳过 |

（`DSV41_E2E_LAYERS=6` 时首次 NaN 仍应落在 L3/L4；若 6 层没复现，用 40 层再试 —— 漂移是概率事件。）

### 8.2 真实 KV 的 40 层单进程链（从未跑完的那次）

```bash
cd /opt/dlami/nvme/saboxu/mirage/demo/deepseek_v4_1_sglang
export PATH=/opt/dlami/nvme/saboxu/venvs/mirage/bin:$PATH
export LD_LIBRARY_PATH=/opt/dlami/nvme/saboxu/venvs/mirage/lib/python3.12/site-packages/z3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/dlami/nvme/saboxu/mirage/python:/opt/dlami/nvme/saboxu/mirage/demo
# 注意：MIRAGE_STATIC_BUILD_CACHE 是**目录**，不是开关。早期脚本里写成 `=1` 就在 demo 目录下
# 造出一个叫 `1` 的 build 缓存（80 层 nvcc 产物，已加进 .gitignore）。别设它，或设成一个真目录。
python -u full_decode.py \
  --checkpoint /opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017 \
  --inputs-dir /tmp/prefill_row0 --golden golden_bench32k.jsonl --row 0 \
  --steps 0 --rms --timeout 240 \
  --save-logits /tmp/logits_row0.f32 --out /tmp/full_real \
  2>&1 | tee /tmp/fullreal2.log
```

注意 `--out /tmp/full_real`（**已有 80 个 build**，不是新的）。

### 8.3 真正的验收（goal 指定的那条链）

**用 `l2_golden.py`，不是 `golden_all.py`。** `golden_all.py` 只跑 stock SGLang（生成 golden 本身）；
验收需要的是**同一个脚本、同一 prompt、全 40 层静态**，而 `l2_golden.py` 正是这个：
它走真实 32K prefill、`DSV41_STATIC_*` 生效、写出与 golden 完全同 schema 的 jsonl。

一键跑完（`accept.sh`）——产出物、对照、判定都在里面：

```bash
cd /opt/dlami/nvme/saboxu/mirage/demo/deepseek_v4_1_sglang
./accept.sh                 # row 0, 32 greedy steps
```

它做的四件事：

| # | 命令 | 产出 | 意义 |
|---|---|---|---|
| 1 | `l2_golden.py` + `DSV41_STATIC_ATTN=all MOE=all POOL=1` | `/tmp/accept/static_row0.jsonl` | **被测对象**：40 层全静态 |
| 2 | `l2_golden.py`，hook 关掉（无 `DSV41_STATIC_*`） | `/tmp/accept/baseline_row0.jsonl` | **噪声底对照**：同一 engine、同一 row，只是不用静态核 |
| 3 | `compare_golden.py --golden golden_bench32k.jsonl --other baseline_row0.jsonl` | `baseline_vs_golden.txt` | engine 自己的漂移 → 这才是 `--tol` 该用的数 |
| 4 | `compare_golden.py --golden golden_bench32k.jsonl --other static_row0.jsonl` | `static_vs_golden.txt` | **验收** |

外加一条最直接的对照：`compare_golden.py --golden baseline_row0.jsonl --other static_row0.jsonl`
—— 把 engine 噪声完全消掉，剩下的差异只可能来自静态核。

判据：`--tol 0.15`（实测噪声底约 0.125）、`--min-overlap 0.6`；argmax flip 只报告、不判失败。
**NaN 与 sink 现在是硬失败**（见下）。

#### 验收门本身修过一个洞（重要）

`compare_golden.py` 原来**会让 NaN 通过**。文档写着 "value ... is the check that fails on NaN"，
但 `nan > tol` 在 Python 里是 `False`，而且 `max()` 会把后出现的 NaN **丢掉**（`nan > x` 为 False，
running max 保留有限值）。实测：最后两步全 NaN、token 序与 golden 相同的一行，输出
`all rows within tolerance`、**exit 0**，`maxval` 还显示 0.000。这正好是 objective 要抓的那个失败形态。

已修（`3be9901c`）：`isfinite` 判非有限并报步数；连续 ≥8 步 argmax 恒定而 golden 在变 → 判 "collapsed"；
表格新增 `id0 n/steps` 让**部分** sink 也看得见。三个构造用例都验过
（相同 → 0；末尾两步 NaN → 1；有限 collapse → 1），`--self` 对真实 golden 回归不变
（163 行 / 5216 步 / 0 flips / `id0 0/32`）。

#### 两个会白烧一整轮的陷阱（都已踩过并堵住）

1. **`boot/sitecustomize.py` 的加载白名单原来没有 `l2_golden`。**
   它只在命令行匹配 `("spawn_main", "sgl_generate", "sgl_gsm8k", "one_batch", "l2_check", "golden_forced")`
   时才 import hook。`l2_golden.py` 的文档却明确要求带 `DSV41_STATIC_*` 跑它，
   而名单里没有 → **父进程不会被 patch**（TP worker 靠 `spawn_main` 侥幸匹配到）。
   已验证干净的那条链（`l2_check`，白名单内）实测有 9–10 条 `patched SGLang`、8 条
   `static layers serving`。为让 `l2_golden` 与之一致，已把 `"l2_golden"` 加进名单。

2. **`env` 不带 `-u` 会继承当前 shell 的 export**，于是"baseline"可能悄悄也跑静态核 ——
   两个 run 一样，对照白做，而且**两边都会"通过"**。`accept.sh` 现在对 baseline 显式
   `-u DSV41_STATIC_ATTN -u DSV41_STATIC_MOE -u DSV41_STATIC_POOL -u DSV41_STATIC_CHECK`，
   并且**用日志自证**：static run 必须 ≥ `SGL_TP` 条 `patched SGLang`，baseline 必须 **0** 条，
   否则返回 9 而不是继续。

（`_ids("")` 返回**空集**、`None` 只对应 `"all"`，所以"unset = 不介入"本身是对的；
问题只在上面两处传递链。）

再做两件"证明门槛没定错"的事：

1. **噪声底对照**：上面第 3 步就是它（比 `bisect_repeat.sh` 更直接：同一 row、同一 engine 配置）。
   要更严格的两个全新 engine 版本，用 `bisect_repeat.sh`。
2. **验收工具自检**：`python compare_golden.py --golden golden_bench32k.jsonl --self`
   → 已实测通过：163 行 / 5216 步，`maxval` 全 0.000、`minovl` 全 1.00、0 flips。
   这证明 loader 能吃下整个 golden 文件（`error` 行会被跳过）。

---

## 9. 变更记录

| 文件 | 内容 |
|---|---|
| `demo/deepseek_v4_1_sglang/FINDINGS_correctness.md` | 本文 |
| `include/mirage/static_megakernel/core.cuh` | `STATIC_TP_BARRIER`：内核内跨 rank 起点栅栏（原本是死代码，现由 build 定义）；`DSV41_RESET_DEBUG` 的 reset 记账 |
| `include/mirage/static_megakernel/host.cuh` | `@prime`：重新武装 by_reader 缓冲区与 stamps；`DSV41_RESET_DEBUG` 的 reset 列表 dump |
| `include/mirage/static_megakernel/runtime.cuh` | `DSV41_MBAR_DEBUG`：有界等待并报告未见到的 mbarrier |
| `include/mirage/static_megakernel/tasks/{gemm_fp8_tile,sparse_attn}.cuh` | `DSV41_SA_QPOLL_DEBUG`：有界等待并打印轮询到的字节 |
| `demo/deepseek_v4_1_sglang/dsv41_static_layer.py` | `tp > 1` 时默认加 `-DSTATIC_TP_BARRIER`（`DSV41_STATIC_TPB=0` 关闭）；`DSV41_STATIC_SERVING=0` 走 host reset 路径 |
| `demo/deepseek_v4_1_sglang/dsv41_static_hook.py` | `TPBAR` + `_tpb`；banner 打印 `TPBAR=` / `TPB=`；两者皆 0 且纯静态链时打 WARNING |
| `demo/deepseek_v4_1_sglang/full_decode.py` | 单进程 40 层链；mHC `S0 = embed.repeat(HC)`；logits 取 rank 0 全量 |
| `demo/deepseek_v4_1_sglang/dump_hook.py` / `dump_prefill.py` / `dump_boot/` | 从 TP worker 内部 dump prefill 的 KV pool / 寻址表 / 每层残差 |
| `demo/deepseek_v4_1_sglang/l2_check.py`、`golden_all.py`、`golden_set.py`、`golden_forced.py`、`compare_golden.py`、`e2e_decode.py`、`bisect_repeat.sh` | 逐 sublayer 对照、golden 生成与比对、噪声底 |
| `.gitignore` | `demo/deepseek_v4_1_sglang/1/`（`MIRAGE_STATIC_BUILD_CACHE` 是目录，被误写成 `1` 的产物） |
