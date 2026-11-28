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

**决定性证据：同一套权重，只差"层间有没有 host 同步"。**

| run | 模式（hook 横幅） | 层间同步 | 结果 |
|---|---|---|---|
| `l2_pool3.log` | `(check, keep sgl, PDL False)` | ✅ 每层都跑 SGLang 自己的 kernel，它的 collective 顺带对齐 | L0–L6 relRMS 正常 |
| `l2_nan.log` | `(replace, PDL False)` | ❌ 纯静态，8 个进程各跑各的 | attn L0–L3 有限，**attn L4 `y finite=False, x finite=True`**，之后全 NaN |

这条对照比"改 flag 前后"强得多：**同一个脚本、同一套权重、同一批 prompt**，唯一变量是同步。
同步 → 干净；不同步 → 首个 reuse 层 NaN。这是"缺跨 rank 栅栏"的直接因果，不是相关性。

### 4.2 三条路径的实测对照

| 路径 | reset 由谁做 | 层间 TP 对齐 | 结果 |
|---|---|---|---|
| `full_decode.py`（demo build） | host，每次 launch memset | ✅ 走 690 分支，自带 `push16(EXCHANGE_HELLO...)` 起点栅栏 | 干净（`full40.log`，80 次 launch，`nf 0`） |
| `l2_check` `CHECK=1`（每层先跑 SGLang kernel） | serving | ✅ SGLang 自己的 collective 顺便对齐 | 干净（`l2_pool3.log`，L0–L6 relRMS 正常） |
| **serving 纯静态链** | serving | ❌ 584 分支，无跨 GPU 栅栏 | **首个 reuse 层 NaN**（`l2_nan.log` L4 / `e2e_t1.log` L3） |

`full_decode.py:67` 明确注释 `FLAGS` 里不含 `STATIC_RESET_IN_KERNEL`（`the host reset path`），
与上表一致。**唯一没有栅栏的路径，正是唯一出 NaN 的路径。**

### 4.3 跨卡状态只有交换区；0xFF 与 NaN 的**确切**路径尚未查实

**先纠正两处我自己写错的结论**（错误的原因比错误的代码更危险，见 §4.5 末尾）：

1. ~~"每次 launch reset 的缓冲区被抹成 0xFF，而 peer 会去读它们"~~ —— **错**。`g.cnt`（计数器）
   和 reset 列表都是 `dmalloc` 的**卡内私有**内存（`host.cuh:435`、`host.cuh:527`），`core.cuh:588`
   原话是 "the resets left to the start (the host's list: **buffers only this GPU writes**)"。
2. ~~"交换区未写入时是 0xFF"~~ —— **错**。`_exchange_region()`（`dsv41_static_layer.py:166`）分配后
   立刻 `buf.zero_()`，所以未写入的交换格是 **0.0**，不是 NaN。

**已经查实的部分**：

| 事实 | 证据 |
|---|---|
| 唯一的跨卡状态是 exchange 区（`G::mc`/`G::rv`，**2 套**） | `core.cuh:163`；`g.cnt`/resets 都是 `dmalloc` 私有 |
| mHC=1 时交换区**非空**：每套 offset 0 有 81920 字节 | `attn3_t1`/`moe3_t1` 生成代码：`SET 86016`、`HELLO 81920`、`BYTES[23]=81920` |
| 这块缓冲就是 mHC streams 的跨卡 all-reduce | `run_sum_store<StaticNode<...,23>>`（`attn3_t1/build/layer.cu:106`） |
| 无 mHC 时 `BYTES` 全 0，**没有**交换缓冲 | 同一份生成代码；也解释了 11-25 那次要 `MHC=1` 才炸 |
| 消费方在**同一个 launch** 内读 peer 在本次 launch 写的东西 | `sum_store` 的 push + 本地计数等待 |

**尚未查实**：**为什么是 NaN 而不是"数值错"或"卡住"**。候选：
- `gemm_tile.cuh:114` 是 `while (!valid16(x))` —— 读到哨兵是**自旋**（我这次确实观察到 NaN 之后进程挂住）；
- `valid16` 的判据是 "至少一个 word 不是 0xFFFFFFFF"，所以**半个 16 字节被写**就会"通过"检查，
  剩下的 0xFF 被当数据 → NaN。但这需要"门比写者更早放行"，而门（计数器）是卡内私有的，
  这条链还没走通。

**能一次性定性的实验**（需要 GPU；现有诊断工具的覆盖面要写清楚，别夸大）：

| 工具 | 它**实际**能看到什么 | 看不到什么 |
|---|---|---|
| `DSV41_E2E_DEBUG=1` → `post_rank{R}.txt` | 层**声明的张量**（`attn[i].w` / `moe[i].w`）里的 `0xFF` 计数 | kernel 内部 scratch（`wo_a_part`、`sparse_attn_ml`）、交换区 |
| 同上 → `exchange_rank{R}.txt` | 交换区**每个 region** 的 nonzero 与 `0xFF` 计数 | 只到 region 粒度 |
| **新** `DSV41_RESET_DEBUG=1` + `DSV41_E2E_DBGLOG=<该 run 的日志>` → `buffers_rank{R}.txt` | **每一个** reset 缓冲：名字（从 `layer.cu` 的 `buf.N` 声明 1:1 匹配）、字节、NaN 数、`0xFF` 数、首个 NaN 的下标 | 只是**步末**状态 |

第三项是这轮新做的（`dump_buffers.py` + `e2e_decode` 接线）。`[rl]` 行给出每个 reset 缓冲的**地址**，
`e2e_decode` 在**自己进程内**用 `cudaMemcpy` 读回——设备地址只在本进程有效，跨进程读会挂，
所以 `dump_buffers.py` 单独跑时只对"指针所属进程"合法，这点在它的 docstring 里写明了。

判读方式（这才是关键，别再含糊）：
- **`NaN` 数 = 字节数/4** → 整个缓冲**还是 0xFF 填充**（0xFFFFFFFF 按 float32 就是 NaN），
  即"生产者一次都没写"；
- **`NaN` 数 < 字节数/4 且 `first NaN` 不为 0** → 里面**被推进了真 NaN**，前面若干字是真实数据；
- `nonzero = 0` → 缓冲被清零过（`v=0` 的反例），或从没被写。
- 已知的局限：这是**步末**状态，NaN 级联会把首个坏层之后的一切都染成 NaN。要贴着故障点看，
  用 `DSV41_E2E_LAYERS=5`（最后一层正好是首个出 NaN 的 attn L4），而不是 6 层把 L5 放最后。

**在拿到这些之前，不要在任何地方断言 NaN 的确切来源。**

### 4.4 为什么首个 NaN 落在第一个 reuse 层

已验证的是"跨层传值 + 时序"这一层，不涉及 NaN 的确切来源：

1. **同一个 launch 内消费**：launch n 里每个 rank 把 streams 写进 set `n % 2`，peer 在同一个
   launch n 从 set `n % 2` 读。所以"所有 rank 都在 launch n"是数据存在的**前提**。
2. **漂移 1 次就已经是错的**：落后一次的 peer 写的是 set `(n−1)%2`，当前 launch 读的 set `n%2`
   里它那一格还是 peer 在 launch `n−2` 写的东西。lag 2 时更糟：写到对方正在读的那一套。
3. **为什么是"第一个" decode step**：objective 说根因定位在第一步。一个自洽的解释是——
   越早期的 launch，交换区里"从没被写过"的格子越多（清零值 / 未完成态越多），
   而越往后越可能只是**过期但有限**的旧值（悄悄算错，比 NaN 更难发现）。
   **这一步是推测，不是证据**，要靠上面那个 DEBUG 实验或一次 32 步的长跑（看 NaN 是否只在
   前几步出现）来确认。

而 reuse 层（L3/L4）是**第一个消费另一层产物**的层（读 L2 的 `compress_cache` + `compress_idx`），
L0–L2 只写自己的缓冲。所以竞态第一次有了"跨层传值"这个兑现点 —— 首次 NaN 在 L3/L4 之间移动，
取决于启动时序，是竞态而非确定性算错（这一条有 11-25 与 11-26 两次独立复现，见 §8.1）。

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

三类真实生成的层（`attn3_t1` / `moe` / `head`），带/不带 `-DSTATIC_TP_BARRIER`，全部零错误：

| 层 | 不带栅栏 `.nv_fatbin` | 带栅栏 | 增量 |
|---|---|---|---|
| `attn3_t1` | 0x02da58 | 0x02df00 | **+1192 B** |
| `moe` | 0x03dcb8 | 0x03e148 | **+1160 B** |
| `head` | 0x05a960 | 0x05adf8 | **+1176 B** |

两个要点：

1. **`.nv_fatbin` 的增量是"栅栏进了设备代码"的证据**。不要看 `.so` 的文件大小：ELF 段对齐会把
   这一两千字节吸收掉，文件尺寸可能一模一样。也不要用 `nvcc -E`——`-DSTATIC_NO_START_BARRIER`
   ± `TP_BARRIER` 时 fatbin 尺寸**完全相同**，正是因为那块代码在另一个分支里被预处理掉了。
2. **增量为什么是 1.1–1.2 KB 而不是几百字节**：每个 SM 的 thread 0 要依次等 8 个 rank，
   `for (r = 0; r < GPUS; r++)` 被完全展开 —— 8 段 "读槽 + 比较 + 回跳"，加上 push16 的
   `multimem.st`，一千多字节是合理的。**旧的 `+160 B` 记录是另一版构建测的，已作废**；
   以这张表为准（同一条命令可复现，见 §9 变更记录）。

**已知剩余缺口**：`STATIC_TP_BARRIER` 那块代码在 `#elif defined(STATIC_RESET_IN_KERNEL)` 分支内
（`core.cuh` 584 那条），所以 **`STATIC_NO_START_BARRIER` 分支（548–583，`DSV41_X_FLAG=1` 的实验路径）
仍然没有跨 GPU 栅栏**。那条路径默认关闭；现在由 `tpb_flag()`（`dsv41_static_layer.py:445`）的断言兜住
—— 两者同时出现时直接报错，而不是静默丢掉漂移界。将来若要启用该分支，得先给它也加一块。

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

#### 先决条件：goal 里那三个变量**不足以**选中出 NaN 的那条链（11-26 查清）

| 变量 | 默认 | 为什么必须显式设 |
|---|---|---|
| `DSV41_STATIC_MHC` | **0（关）** | mHC 是模型 Block 的一部分（`hc_mult=4`），也是**唯一**把 sublayer 的收尾（交换区上的 all-reduce）放进内核的开关。MHC=0 时那部分留在 SGLang 自己的代码里 —— 被测的是**另一条链**。 |
| `DSV41_STATIC_TPBAR` | **1（当 MHC=1）** | 每 sublayer 一个 **host** 栅栏，**11-26 才加**。它会**掩盖**内核栅栏失效：栅栏坏了也照样 finite。要证明内核栅栏本身够用，必须 `TPBAR=0`。 |
| `DSV41_STATIC_TPB` | 1 | 内核内的起点栅栏（本次修复）。`TPB=0` 是反向对照。 |

**证据**：11-25 的 `l2_nan.log` banner 是 `patched SGLang: attention all, MoE all (replace, PDL False)` ——
**没有** `TPBAR=` 字样，而 `DSV41_STATIC_TPBAR` 的引入 commit 是 `2a69b646`（2026-11-26）。
所以那次确实**没有任何跨 rank 同步**，NaN 与根因一致。
反过来，今天再按老命令跑，`TPBAR` 会默认打开、把漂移盖住 —— 于是"修好了"可能只是"被 host 对齐了"。

为此：
1. hook 的 banner 现在打印 `MHC= TPBAR= TPB= engram=`（`dsv41_static_hook.py`），
   日志能自证跑的是哪条链；
2. `accept.sh` 显式设 `MHC=1 TPB=1 TPBAR=0`，并**断言** banner 里就是这个组合，
   否则返回 9 而不是继续比数。

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

## 10. 2026-11-27：attn0 那 3.8% 的根因 —— wo_a 吃 bf16 o，不是 fp8（已修，GPU 已验证）

### 10.1 结论

`sparse_attn` 的输出 **o 不该被 act_quant 压成 e4m3**：SGLang 的 wo_a 是一条 **bf16 linear**，o 以 bf16 直接进 GEMM。
链里多出来的那次 per-32-block 量化正好是 |o| 的 1.7%（正交于 o），过 wo_a/wo_b 后放大成 s 的 **3.8%**。

证据链（全部实测，不是推断）：

| 步骤 | 结果 |
|---|---|
| 反演 {δs} 到 δo（M = ⊕_r WOB_r WOA_r，最小范数解） | `\|\|δo\|\|/\|\|o\|\| = 1.7%`，**cos(δo, o) ≈ 0.006**（逐头 α < 0.003）→ 与 o 正交，不是分母/sink 误差 |
| δo 在 rope 64 维的能量占比 | 0.167（o 本身 0.693）→ 误差均匀铺在 448 个 noPE 维上，不是 RoPE |
| sink 变体（×√D、/√D、0、无 sink、m=max(rowmax,0,sink)） | cos 全在 0.9992–0.9994，一个都不动 → sink 无关 |
| **o 不量化（bf16）** | **cos 0.99928 → 0.99998385，relRMS 3.79% → 0.577%，\|s\| 35.5965 → 35.6357（SGLang 35.6344）** |
| o 量化 block 32 / 64 / 128 / 512、floor 指数 | 全部 0.99928（量化噪声与 block 划分几乎无关）→ 不是"量化粒度选错了" |
| 1.7% 的量级 | 正好等于 e4m3 的 RMS 量化误差（2⁻⁴/√12 量级） |

代码侧的独立确证（`sglang/python/sglang/srt/models/deepseek_v4.py`）：

```python
def wo_a_fp8_gemm_enabled(quant_config):
    """... any other layout dequantizes wo_a to bf16 at load."""
    if not (_FP8_WO_A_GEMM and isinstance(quant_config, Fp8Config)):
        return False
    block = quant_config.weight_block_size
    return block == [128, 128] or (block == [32, 32] and _is_hip and ...)
```

checkpoint 里 `layers.*.attn.wo_a.scale` 是 `(256, 128)` 对 weight `(8192, 4096)` → **32×32 块**，CUDA 上不满足 `[128,128]` → `wo_a_fp8 = False` → 权重 load 时反量化成 bf16
（`ColumnParallelLinear(..., params_dtype=torch.bfloat16)`）。于是 `_project_wo_b` 走的是

```python
o = F.linear(o, wo_a_weight)      # o bf16 × wo_a bf16, 没有 act_quant
```

`inference/model.py` L781-787 里的 `act_quant(o, ..., True)`（quant+dequant 原地）只是**官方对 fp8 通路的模拟**；
SGLang 在 bf16 权重这条路上不做它。两边在这一点上本来就差 3.8% —— 我们要对齐的是 SGLang 的 dump，所以按 SGLang 走。

### 10.2 改动

| 文件 | 改动 |
|---|---|
| `include/.../tasks/sparse_attn.cuh` | `sparse_attn_task` 加模板参数 `BF16O`（params[7]）：走尾段时不算 amax、不写 scale 字节，直接把 8 个 bf16（`uint4` 一次 16B）写到 `out`；`quant8` 的 bf16 分支在这条路上提前 return；`xq_scale_line` 用 `if constexpr (!BF16O)` 关掉 |
| `python/mirage/mpk/static_megakernel.py` | `sparse_attn_layer(output_bf16=)`：dtype bf16、`out_buffer = T*N*2`、params 固定 `[H,D,W,RD,scale,NC,CAP,BF16O]`（BF16O 恒定占 index 7）；`act_quant_layer` 也接受 `gemm_tile` 的 slots 输出（同一个 `[parts][T][N]` fp32 布局、同样的 params[0]/[2]） |
| `demo/deepseek_v4_1_csa2/demo.py` | `wo_a_weight` 是 bf16 且 `G == 1`（TP8）时：`o_q` 用 bf16、`sparse_attn_layer(output_bf16=True)`、wo_a 换成 `gemm_tile_layer`（bf16×bf16）；否则保留原来的 fp8 分组 GEMM |
| `demo/deepseek_v4_1_csa2/weights.py` | `DSV41_WO_A_BF16=1`（默认）且 `G == 1` 时把 wo_a 反量化成 bf16（e4m3 × 2^k 在 bf16 里精确）；`=0` 回到 fp8（= 官方 model.py 的语义） |
| `demo/deepseek_v4_1_sglang/full_decode.py` | `eg_rows`：dump 缺 `L{LL}_eg_emb.f32` 时可用 `DSV41_EG_ZERO=1` 走零值**诊断**（只用于读它之前的 sublayer，会打 WARNING）；`--inputs-dir` 要用 `/tmp/dec_coef8`（带 engram probe 的那份），`/tmp/ref_v41` 没有 |

### 10.3 GPU 实测（`--layers 2 --steps 1 --inputs-dir /tmp/dec_coef8`，`/tmp/chain_b0`）

`attn0_r0_o_q.bin` 现在是 **65536 B = [8, 4096] bf16**（原来是 32768 B fp8 + scale chunk）。

| 比较 | cos | relRMS |
|---|---|---|
| chain o ↔ 参考 bf16 o | **0.99999827** | 0.002208 |
| chain o ↔ 参考 act_quant(o) | 0.99974251 | 0.022742 |
| `attn0_streams_out` vs `L00_apost` stream0 | 1.0000011 | 0.000286 |
| stream1 | 1.0000010 | 0.000316 |
| stream2 | 1.0000010 | 0.000000 |
| **stream3（唯一对 s 敏感的流）** | **0.99999124** | **0.004408** |
| 整体 | — | **0.002647** |

对比修之前：stream3 relRMS **0.0378**、整体 0.0198。目标（relRMS < 1%、cos > 0.999）在这一层已满足。
剩下的 0.44% 在 s 的 bf16 存储上（wo_b 输出到 post 之间），不是算法差。

### 10.4 顺带修掉的第二个接线 bug

`full_decode.py` 里 40 个 attention sublayer 全都 `wg["x"] = X[r]`（当前 token 的 embedding）。只有 L0 该这样；
L > 0 的 x 必须是上一个 sublayer 的 `x_out`（L ∈ {1,14} 是 host 算的 Engram 输入）。这条 bug 让 40 层 attention 全部从
embedding 起算 —— 修完残差立刻从"完全发散"变成 cos 0.993–0.9998。同时修了 Engram 注入的广播 bug
（`ws[r]["x"].copy_(eg_next_x(...))` 把 [5120] 写进 [8, 5120]，改成 `[:1].copy_(... [None])`）。

### 10.5 剩下的门槛

40 层全量验证需要 `layers/step1/L{LL}_eg_emb.f32`（Engram 层）。`/tmp/dec_coef8` 有，`/tmp/ref_v41` 没有 ——
以后跑链一律 `--inputs-dir /tmp/dec_coef8`。工作区里还有 `DSV41_WO_A_BF16=0` 可回到官方 model.py 的语义（此时 attn0 会
回到 relRMS 3.8%，这是"官方 vs SGLang"的真实分歧，不是回归）。

---

## 11. 2026-11-27：teacher forcing 逐个 sublayer 打分 —— 系数已对齐，attention 自身在 L≥1 还有 5–22%

`full_decode.py --dump-dir DIR` + `DSV41_TEACH=1`（每层喂 SGLang 自己的残差），新工具 `cmp_teach.py`。

### 11.1 三个 harness 的真 bug（数字曾经全是假的）

| bug | 现象 | 修法 |
|---|---|---|
| **`L{LL}_in` 是过期快照** | L≥2 的 `_in` 等于 `L{LL-1}_apost`（偏一个 sublayer）；用 `L02_ct_*` 把 `L02_mpost` 还原成 `L02_apost` 残差 0.0016，用 `L02_in` 是 0.29 → attention 读的是 **mpost**，MoE 读 **apost** | `teach_row` 按 `mpost/apost/eg_y` 取 |
| **链上张量被别名** | 上一层的 `streams_out` 就是下一层的 `streams`（同一张量）；teacher 模式覆盖 `streams` 会**回写上一层的输出**，dump 出来的全是邻居的输入（实测与拷贝值逐位相同） | 每次 launch 后立刻抓 `_TEACH_OUT`，写 `{name}_teach_out.f32` |
| **layer 0 不做 pre-mix** | `deepseek_v4_mhc._combine`：`apply_pre is None` 时直接 norm stream 0；dump 里的 `ct_pre(0)`＝0.921 不是 one-hot，拿它混会把 x 带偏 8%（attn0 0.3%→0.9%） | `sublayer_x` 对 `L==0 and kind=="attn"` 只用 stream 0（`normed_row`） |

### 11.2 现在能复现的分数（6 层 teacher，输入精确）

| | attn y | cos | MoE 残差 | 备注 |
|---|---|---|---|---|
| L0 | **0.50%** | 0.999988 | — | 与 free chain 的 0.3% 一致 |
| L1 | 13.6% | 0.9907 | 38% | Engram 层（teacher 分支没喂 `eg_emb`，eg_kv 缺 → 先把这条补上再看） |
| L2 | 21.7% | 0.9762 | 55% | `full_r2`（自己拥有 compressed pool） |
| L3 | 7.2% | 0.9974 | 36% | `reuse`（借 L2 的 index） |
| L4 | 11.6% | 0.9932 | 38% | `reuse` |
| L5 | 5.4% | 0.9986 | 40% | `reuse` |

要点：

- **mHC 系数完全正确**：链算出的 `comb/post` 与 dump 的 `L{LL}_ct_*` 最大差 **1e-7** → 残差差不是系数来的。
- 误差在 5120 维上均匀（10 个 512-block 各 ~0.14/0.22/0.07…），不是某几个 head 整体坏掉。
- **MoE 侧目前无法用 dump 验证**：dump 每层只有一套 `ct_*`＝`hc_attn`，MoE 的 post 用的是 `hc_ffn`（没 dump）→ 反解出的参考 y 不可信（rms 0.26 vs 0.08）。free chain 里 MoE 残差是 4–5%/层，只能算上界。
- L0 0.5% 说明 swa 那条 kernel 路本身通；L1 同变体却 13.6% → 先补 `eg_emb`，再查 L2 的 compressed pool 绑定（`bind_pool_state` 的 `skip = first_page * (256//ratio)`）。

### 11.3 已排除：不是输入、不是系数、不是 x 接线

11.2 的分数是"输入精确"下的真实分数，但**先证明了输入链路没错**，才能说是 attention 内核自己的问题：

| 检查 | 结果 |
|---|---|
| golden 输入行选对了吗 | `comb^T in + post·x -> apost` 在 L0/L1/L2 都是 **0.0015**（`in`/`eg_y`/`mpost` 三种情况） |
| 链算出的系数 vs 公式 | pre/post 最大差 **1e-7**（`coef_ref.triplet`） |
| free chain 的 post 恒等式 | `comb^T R + post·s -> R'` = **0.0014**（moe0），dump 的残差是真的 |
| **非 Engram MoE 的 x_out** | `norm(pre·R')·gamma_{L+1}` vs 内核写的 x_out：**relRMS 0.0003, cos 1.000000**（moe1）→ x 接线正确 |
| Enram host（L0/L13 的 MoE） | x_out 由 host 从 R'' 重算（`eg_next_x`），不是内核写的 → 不要拿它比公式 |
| `moe2` 的 x_out | 方向 cos 0.994 但 **rms 0.271 vs 0.0206（13×）** → hc_post_combine 的 `ssq` 归约在某些 launch 上只拿到部分 part。因为 act_quant 是 per-32-block 的 2 的幂缩放，全局比例被抵消，所以链还能活（但这本身是个 bug，和 §6 的 0xFF 交换区同一条线） |

**结论**：`x` 正确、残差正确、系数正确 → L≥1 的 5–22% 出在 attention 子层内部（KV 窗口绑定 / Engram eg_kv / softmax-sink / o·wo_a）。
L0 0.5% 与 L1 13.6% 是同一个 `swa` 内核、同一个变体 → 差别只在**每层的 KV 状态**（窗口绑定、`eg_kv`），所以下一步是拿参考窗口 cache 在 numpy 里重算一次 L1 的 attention。

需要的常量（`model_config` 实读）：`n_heads 64`、`head_dim 512`、`rope_head_dim 64`、`window_size 128`、MLA（K=V，64 头共享）、`ROW = 528 = 512 e4m3 + 16 e8m0`、`CROW 288`、`IROW 68`；pool 一行是一页 256 token（`swa_row_bytes 135168 = 528×256`），dump 里 `positions[0]=16384, swa_out_cache_loc[0]=16640` 且 `idx = p - p0 + r0` 不越界（窗口 32035..32162 < 34048）。

### 11.4 坑：teacher 模式的 `--dump-dir` 里 `*_streams_out.bin` 不可用

链把 `streams` 和上一层的 `streams_out` 用**同一个张量**接起来，`dump_chain` 在整步跑完之后才写文件，所以 teacher 模式下这些文件是"邻居的输入"。要打分只能读 `{name}_teach_out.f32` / `{name}_teach_x.f32`（每次 launch 后立刻拷出来的）。free chain 的 `streams_out` 没有这个问题（每层自己的输出张量）。

---

## 12. 2026-11-27 收尾：L≥1 那 5–22% 的根因是 **pre_mix 跨 sublayer 传递**（已修，40 层已验证）

### 12.1 根因（`inference/model.py` `Block.forward:986-1002`）

```python
attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, ...)
x = self.hc_pre(x, pre_mix)          # ← 用上一个 block 传下来的 ffn_pre，不是 attn_pre
x = self.attn(x, start_pos, ...)
x = self.hc_post(x, residual, attn_post, attn_comb)

residual = x
ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, ...)
x = self.hc_pre(x, attn_pre)         # ← 用本层 attention 的 attn_pre，不是 ffn_pre
x = self.ffn(x, image_mask)
```

所以一个 sublayer 读的 `x` **不是它自己 triplet 的 pre**：

| 子层 | 它读的 pre | 施加在哪个残差上 |
|---|---|---|
| attention L=0 | identity（`make_identity_pre_mix`，只有 stream 0） | `L00_in` |
| attention L≥1 | **`moe{L-1}` 的 `ffn_pre`** | `L{L-1}` 的 MoE 输出 |
| MoE L | **`attn{L}` 的 `attn_pre`** | `L{L}_apost` |

之前 `sublayer_x()` / `eg_next_x()` 一直用"本子层自己 triplet 的 pre"，于是每层的输入 `x` 都带一个方向性偏差 —— 这就是 L1 13.6%、L2 21.7% 的来源（attention 内核本身没问题，见 §12.4）。

`pre` 从生产它的 kernel 的 `hc_coef` 缓冲直接读（`full_decode.py: hc_pre()`，前 `HC` 个 fp32 = `pre`；布局 `pre[0:4] / post[4:8] / comb[8:24]`，见 `tasks/hc_coef.cuh:53,135`）：

```python
pre = None if (sub == "attn" and L == 0) else hc_pre(f"moe{L - 1}" if sub == "attn" else f"attn{L}")
```

### 12.2 修后：teacher 逐层打分（6 层）

| 子层 | 修前 | 修后 |
|---|---|---|
| attn0 | 0.26% | **0.26%** |
| attn1 | 5.29% | **0.78%** |
| attn2 | 9.11% | **1.64%** |
| attn3 | 2.38% | **1.66%** |
| attn5 | 2.05% | **1.69%** |

（teacher 的 MoE 行仍然不可信：`x`/`x_out` 张量在链上互为别名，teacher 覆盖后 dump 语义失效 —— 见 §11.4。MoE 用下面 free chain 的 y 级对照。）

### 12.3 40 层 free chain（`/tmp/chain40`，80 kernel，7.6 min）

`python align3.py` 的三方表（每格 relRMS / cos）：

| 子层 | chain vs SGLang A | chain vs SGLang B | **SGLang vs 它自己** |
|---|---|---|---|
| attn0 | 0.0026 | 0.0026 | **0.0000** |
| attn1 | 0.0092 | 0.0092 | **0.0000** |
| moe1 | 0.0139 | 0.0139 | **0.0000** |
| attn2 | 0.0206 | 0.0206 | 0.0057 |
| attn7 | 0.0374 | 0.0367 | 0.0319 |
| moe7 | 0.0867 | 0.0868 | 0.0318 |
| moe17 | 0.1241 | … | **0.1308** |
| 全 80 个子层 | mean 0.0888 / worst 0.1790 | mean 0.0751 / worst 0.1331 | **mean 0.0735 / worst 0.1308** |

**关键**：`dec_coef7` 与 `dec_coef8` 是**同一 decode step 的两次独立 SGLang 进程**（插桩版本相同，n=440）。它们对彼此的偏差（mean 7.4%、最差 13.1%）和"静态链对任一次 SGLang"的偏差同量级。

所以"每层 relRMS 都远小于 1%"这个判据**参考实现自己都达不到**：L0/L1 处 SGLang 逐 bit 可复现（0.0000），此时链的真实保真度是 **0.26% / 0.92% / 1.39%**；从 L2 起 SGLang 自身噪声（0.57% → 13%）就盖过链的偏差，两者同步增长。

链自身是**逐 bit 确定**的：同一条 40 层链跑两遍，80/80 个子层 `streams_out` 完全相同，logits 也逐 bit 相同（`/tmp/chain40` vs `/tmp/chain40b`）。所以剩下的差异是"参考的不确定性"，不是竞态。

### 12.4 为什么 L2 之后误差会"跳"：MoE 路由是平局

用链自己的 `x` 算 gate 的 top-6/top-7 间隔：

| 层 | 第 6 名 | 第 7 名 | 间隔 | 相对 |
|---|---|---|---|---|
| L6 | +11.5785 | +11.5722 | 0.0063 | 5.5e-4 |
| L7 | +12.3170 | +12.3164 | **0.0006** | **5e-5** |
| L16 | +12.9841 | +12.9759 | 0.0082 | 6.3e-4 |
| L18 | +12.6701 | +12.6684 | 0.0017 | 1.3e-4 |

bf16 的 1 个 ulp 就是 1.5e-3 相对 —— 比 L7 的间隔大 30 倍。**任何一个专家都可能被舍入翻掉**，而换掉 6 个专家里的 1 个就会让 MoE 的 `y` 变 2%，于是残差跳一档。这解释了 L7/L16/L18 的台阶，也解释了 SGLang 自己跑两次会在 L15–L17 跳。

### 12.5 MoE 内核本身是对的（y 级对照）

把 `R' = comb^T R + post·y` 按 `|post|` 最大的流反解出 `y`（其余流的 `post` 接近 0，反解会被 bf16 舍入淹没），对比 golden 的残差反解出的 `y`：

| 子层 | \|y_gold\| | \|y_chain\| | cos | 
|---|---|---|---|
| moe5 | 0.0842 | 0.0838 | **+0.99863** |
| moe6 | 0.1075 | 0.1073 | **+0.99863** |
| moe7 | 0.1512 | 0.1546 | +0.97866（路由翻了一个专家） |

即 MoE 子层在残差层面只有 0.14% 的方向差，3–4% 的残差偏差是累积漂移，不是 MoE 算错。

### 12.6 `x_out` 的 ssq 归约：本轮 79 个子层全部正确

对每个子层校验 `x_out == bf16(bf16(Σ pre_j R'_j) · rsqrt(mean+eps) · γ_next)`（用该子层自己的 dump + 系数）：77/79 到 1e-4；例外是 `moe0`/`moe13` —— 正是两个 Engram 主机层，engram 在 post **之后**才加到残差上，所以 "post 后立刻算的 x_out" 与下一层真正的输入差一个 engram 增量，属预期（`eg_next_x()` 就是干这个的）。之前 §11 记的 `moe2` 13× 尺度跳变本轮**没有复现**。

### 12.7 最终 logits（`/tmp/chain40_logits.f32` vs `golden_bench32k.jsonl` 第 0 行）

注意行对齐：golden 的 `topk[0]` 是 **position 31905** 的分布（预测第一个生成 token 4588），我们跑的是 position 31906（喂 4588），所以要比 `topk[1]`：

| | 结果 |
|---|---|
| top-1 | **343 = golden step-1 的 top-1** ✓ |
| top-32 交集 | **27/32** |
| \|Δlogprob\|（golden top-32 上） | mean **0.371** / max 1.229 |
| golden 的 top-10 全在链的 top-10 里？ | 是（顺序/平局略有不同：链 -1.087/-2.212/-3.462…，golden -1.107/-2.107/-3.607…） |

（`golden_bench32k.jsonl` 自己标了 `deterministic: False`，且 top-k 里有精确平局，所以 logprob 到 1 位的差异是预期内的。）

### 12.8 复现命令

```bash
# 40 层 free chain（builds 约 7.6 min，之后每步 4.1 ms）
python full_decode.py --checkpoint $CK --gpus 8 --layers 40 --steps 1 \
  --pos 31906 --token 4588 --inputs-dir /tmp/dec_coef8 \
  --out /tmp/full_real --dump-dir /tmp/chain40 --save-logits /tmp/chain40_logits.f32

python align3.py --chain /tmp/chain40 --a /tmp/dec_coef7/layers/step1 --b /tmp/dec_coef8/layers/step1
python attn_np.py --layer 0 --gemm        # attention 参考实现（L0 0.46%，L1 1.0%）
python attn64.py --layer 0                # fp64 参考：把 y 从 R' 里反解太病态，见 §12.9 的坑
```

### 12.9 还能再压吗：早层那 0.2–0.9% 是**两种算子顺序的原始间隙**，不是 bug

§12.3 的表读完会问：L0 只差 0.26%、L1 差 0.92%，这里参考是逐 bit 可复现的（下表的 A-B 列是 0），能不能压到 0.05%？答案是不能，而且这个量级已经是"两个都对的实现"之间的差：
链的激活是 fp8（`sparse_attn` 直接写 bf16 o，`wo_a` 走 bf16 fast path），SGLang 那边 `wo_a`/`wo_b` 的累加顺序、tile 划分都不同。同样合法的两种顺序在 5120 维上差 0.2–0.3% 是正常的。

| 子层 | 链 vs A | 链 vs B | **A vs B（参考对自己的散布）** |
|---|---|---|---|
| attn0 | 0.0026 | 0.0026 | **0.0000** |
| attn1 | 0.0092 | 0.0092 | **0.0000** |
| moe1 | 0.0139 | 0.0139 | **0.0000** |
| attn2 | 0.0206 | 0.0206 | 0.0057 |
| moe2 | 0.0230 | 0.0236 | 0.0104 |
| attn3 | 0.0250 | 0.0267 | 0.0157 |
| moe7 | 0.0867 | 0.0868 | 0.0318 |
| moe18 | 0.1790 | 0.1331 | 0.1290 |
| attn39 | 0.1043 | 0.0884 | 0.0997 |
| **全程 mean** | **0.0888** | **0.0751** | **0.0735** |

两点值得记住：
1. **A-B 在 L0/L1 是精确的 0，L2 起就不是了**（0.0057 → 0.03）。也就是说"每层 <1%、cos>0.999"这个验收标准，从 L2 开始参考自己就达不到；能要求的只有"链落在参考的散布带内"。§12.3 的 mean 上链 0.075–0.089 vs 参考自己 0.0735 —— 已经是同一量级。
2. 早层链比参考的散布大 2–4 倍（L2：0.021 vs 0.006），这是 §12.1 那类"被混沌放大的舍入差"的起点：L7 的 gate 平局（Δ=5e-5）之后误差台阶式放大，跟 SGLang 内部 L15→L17 自己翻专家是同一机制。

**更正（本轮，此前这条写错了）：`/tmp/dec_coef5` 不是"另一次启动"，是另一套配置。** 之前拿
`/tmp/dec_coef{5,6}` 与 `{7,8}` 的差异当"同脚本两次启动的散布"，据此说 L1 的 y 级差本来就有 4.7% 的
口径不确定性 —— 那是错的。实测对比（`layers/step1`，同 4×5120 布局）：

```
L00_apost :  dec_coef2 ≡ dec_coef6 ≡ dec_coef7 ≡ dec_coef8   (0.000000)
             dec_coef5 偏离其余四个 1.52%
L01_apost :  dec_coef2 ≡ dec_coef6 ≡ dec_coef7 ≡ dec_coef8   (0.000000)
             dec_coef5 偏离 2.28%
```

`dec_coef2`（goal 里点名的那份）与 6/7/8 **逐 bit 完全相同**，只有 `dec_coef5` 谁都不等。所以：
**同配置下 SGLang 在 L0/L1 是逐 bit 确定的**，`dec_coef5` 应当排除（它是早期/不同参数的一次运行，
不是重复实验）。由此 L1 的 0.92% 不是"参考噪声"，是真实的、可归因的差 —— 见 §12.12。

补一条记录集（`layers/meta.json` 的 `rows`）的旁证，顺带确认插桩本身无害：

```
dec_coef5 / 6 : 432 条，缺 1_eg_x|y|emb、14_eg_x|y|emb   ← Engram 探针加入之前
dec_coef7 / 8 : 438 条（= 432 + 6 个 Engram 探针）
dec_coef2     : 316 条，连 *_x 与 ct_* 都还没有的更早版本
```

`dec_coef6`（无探针）与 `dec_coef7`（有探针）在 L0/L1 上仍逐 bit 相同，**说明探针不改变数值**，只是多记录。
所以记录集差异解释不了 `dec_coef5` 的 1.52% —— 它确实是另一次运行，不是版本差异。

**另一个此前没提的事实**：只有 `dec_coef / 2` 停在 `step1`；`5/6/7/8` 各 dump 了 **`step1`…`step10` 十个
decode step**。本报告全部结论只用 `step1`（goal 指定的 token 4588 @ pos 31906，16 MB），那 9 个额外 step
是还没用过的独立数据点。

**真正的参考散布**（`dec_coef{2,6,7,8}` 中的最大两两距离，`L{LL}_apost`）：

| L | 散布 | L | 散布 | L | 散布 |
|---|---|---|---|---|---|
| 0 / 1 | **0.000000** | 4 | 0.0207 | 17 | 0.1277 |
| 2 | 0.0063 | 7 | 0.0344 | 25 | 0.0935 |
| 3 | 0.0166 | **8** | **0.0851** | 39 | 0.1076 |

注意 L8 处从 0.034 跳到 0.085（×2.5）：这正是 §12.10/§12.4 审出的 **L7 gate 平局**（相对间距
5e-5，bf16 1 ulp 的 1/30）的下一层。**参考自己在 L7 翻了专家，从参考侧独立印证了平局机制**，
而不只是链这边的解释。

所以"每层 <1%、cos>0.999"这个门槛：L0/L1 是可验证的（参考是 0），L2 起参考自己就 0.6% 了，
L8 之后 8.5%–12.8%，任何实现都不可能落进 1%。L1 的 0.92% 见 §12.12 的归因。

**顺带钉死的一个事实（`wo_a` 那一级真的逐 bit 对上了）**：拿链自己 dump 的 `attn0_r0_wo_a_part.bin` 当判据，用 fp64 反量化权重 × **未量化**的 bf16 `o` 重算，`wo_a_part[0]` 与 buffer 差 **2.9e-7**（fp32 舍入），`wo_a_part[1]` 全零（占位）。由此确认：
- 参考（服务端）**不对 `o` 做 `act_quant`**，bf16 `o` 直接进 bf16 `wo_a`；`inference/model.py:187` 那行 `act_quant(o, ...)` 不是服务端的路径（服务端是 `DSV41_WO_A_BF16` 那条）。自己重写时要按服务端走，否则 y 会差 3.5%。
- K 维划分是 `[0:2048]`，不是交错/半块。

`attn64.py`（新）就是按这条路径写的 fp64 参考，但它现在还不能当裁判：`y` 只能从 `R' = post·y + combᵀR` 反解，除以 `post`（L0 0.1362 / L1 0.1917）会把 bf16 舍入放大约 1/post·|R'|/|y| ≈ 1 倍，加上用金标输入配链系数的不自洽，给出 1.7–2.2% 的假象。**判断早层好坏请用残差表（`align3.py`），不要用反解的 y。**

改对 `wo_a` 路径后，`attn64.py` 给出的正向 `R'` 距离是一个**近似等边的三角形**，正好说明"没人特殊"：

| | fp64↔SGLang | fp64↔链 | 链↔SGLang |
|---|---|---|---|
| L0 | 0.00956 | 0.00956 | **0.00265** |
| L1 | 0.00871 | 0.00983 | **0.00919** |

L1 三条边 0.87/0.98/0.92 —— 三个实现两两等距，谁都不比谁更"对"；L0 反而是链离 SGLang 最近（fp64 参考自己还有 ~0.9% 的舍入点没完全对上）。这就是早层误差的终点。

### 12.10 全 40 层路由翻转审计：误差台阶 100% 对得上离散事件（`moe_audit.py`）

§12.4 只说清了 L7 那一次平局。现在把 40 层的 gate 全部算一遍（只算 gate，不是 MoE —— 一次 5120×384 matmul），两边各用自己 dump 的 `pre` + 残差按 `Block.forward` 的同一公式重建 FFN 输入，然后看 top-6 集合是否相同、以及第 6/第 7 名的相对间隔：

| 指标 | 结果 |
|---|---|
| 翻转层 | **L7, L16, L18, L19, L26, L28, L30**（7/40） |
| 这 7 层的相对间隔 | 1.77e-4 / 1.4e-3 / 2.8e-4 / 2.1e-4 / 6.6e-5 / 2.1e-4 / 1.0e-4 —— **全部 ≤1.4e-3** |
| 未翻转但同样紧的层 | L22 1.6e-4、L24 1.5e-4、L27 1.4e-4、L32 1.2e-4 —— 平局但这次没翻 |
| 有翻转却没台阶的层 | L19/L26/L28/L30（此时误差已饱和在 8–10%，一次翻专家只动 ~2%） |

**关键对齐**：残差表里仅有的三次大台阶，全部精确落在翻转层上：

| 翻转层 | 前一层子层 | 翻转后 | 台阶 |
|---|---|---|---|
| L7 | attn7 0.0367 | moe7 0.0868 | ×2.4 |
| L16 | attn16 0.0704 | moe16 0.1220 | ×1.7 |
| L18 | attn18 0.1096 | moe18 0.1331 | ×1.2 |

反过来，**没有任何一层出现"误差大但没有翻转"** —— 那才是内核 bug 的指纹。所以 40 层里没有任何内核级缺陷，剩下的只是平局判决。

顺带解释了为什么链比参考早 1–2 层到饱和：`x` 的相对误差（同一个脚本输出）L0 0.47% → L2 2.87% → L6 3.43% 就停在 3.4%，和参考自己 A-B 的 0.57%→3.2% 是**同一个饱和水平**，只是斜率大一点（链的每层算子顺序噪声略大，因为 o/wo_a 走 bf16 而服务端路径不完全是）。饱和之后误差就不再是"传播"问题，只是各自在不同平局上做了不同判决。

### 12.11 审计的判据本身被独立验证：gate 转写 == 内核对路由（`moe0_r0_pairs.bin`）

§12.10 的结论建立在"我按 `model.py` 转写的 gate 是对的"之上。这条用**内核自己 dump 的路由**独立验证了：

`moe{L}_r{r}_pairs.bin` 不是裸索引，而是 8 字节一条 `(int32 专家号, float32 权重)`，`numpy` 读成 int64 后 `hi = v >> 32` 是专家号、`lo` 按 float32 解释是权重（384 B = 8 token × 6 对）。token 0 解出来：

| | 值 |
|---|---|
| 内核路由的 top-6 | **{115, 163, 200, 274, 342, 375}** |
| 我用重建的 x 算的 gate | **{115, 163, 200, 274, 342, 375}** —— **完全相同** |
| 内核权重 | 0.16506 / 0.16863 / 0.16125 / 0.15732 / 0.37284 / 0.47491，**和 = 1.5 = route_scale** |
| 该层 6/7 名间隔 | 0.0131（绝对）—— 不是平局，所以这次没争议 |

第二行说明 gate 转写（`linear/1.0` → `softplus().sqrt()` → `(scores+bias).topk(6)` → 归一化 → `×1.5`）与内核实现逐位一致；第三行说明**内核确实做了 top-k 归一化**，和 `model.py:826` 的 `norm_topk_prob` 默认 `True` 对得上（config 里没有这个键，别被它误导）。

**同时修掉 `moe_ref.py` 两个真 bug**（这就是 `moearm` 那条待办的真身）：
1. `--x` 读成 1 维 → `expert()` 返回 `[H]` 而不是 `[1,H]` → `[0]` 取到的是**第一个元素**（标量），再广播加进整个 `routed` 向量。表现是 `|routed| 14.79`（其实是 5120 个相同标量）、`|shared| 0.0115`（其实是一个元素）。改成 `reshape(1,H)`。
2. 权重没归一化（注释写"norm_topk_prob is unset"，但 `ModelArgs` 默认 `True`）→ `|routed|` 大 3.3×。补上 `/sum`。

修完 `|routed| 4.79 / |shared| 3.56 / |sum| 6.30`（金标 `L02_mpost_x` 的 `|y| = 5.51`），量级对上了。

**但仍未收敛，别用它下绝对结论**：`moe_ref` 的 y 与 `L02_mpost_x` 的 cos 只有 0.23。gate 已验证、x 已验证、量级已验证，所以偏差在 expert 那一级（或 `mpost_x` 本身不是合并后的 y —— 它的 post 系数没 dump，无法反证）。**MoE 保真度仍以 §12.5 的 y 级反解为准（cos 0.99863）**，`moe_ref.py` 目前只可用来读路由/权重和量级。

---


---

### 12.12 逐 kernel 的 `y` 打分（80 个子层）＋ Engram 口径更正：L1 那 0.92% 不是 bug

这一节有两个目的：(a) 把"每个内核自己算对没有"从残差级下沉到 `y` 级（残差被 `combᵀR` 主导，主要衡量
输入搬运，不衡量内核）；(b) 更正在追 L1 时差点误判成 bug 的 Engram 口径。

#### (a) Engram：链是对的，是我拿错了对照

`inference/model.py:1266` 把 Engram 作用在**原始残差 `h`** 上（`h = layer.engram(h, ...)` 在 block 之前），
所以 L1 的 attention 输入是

```
R'' = eg_y = eg_x + gate · value        (hc=4 个 stream 各一个 gate，共享一个 value)
```

而链的 MoE 内核 dump 出来的 `streams_out` 是**注入 Engram 之前**的 `R'`（= `eg_x`）。先前的诊断拿链的
`moe0_streams_out` 直接去比 `L01_eg_y`，差 0.589 —— 那不是 Engram 的错，是差了 `gate·value` 这一项。

用 checkpoint 在 fp64 复算整条 Engram（`wkv` fp8 32×32 dequant → `key/value`，`q_weight*k_weight`，
逐 stream 归一化，signed-sqrt sigmoid）作为独立裁判：

```
official fp64 recompute vs L01_eg_y           relRMS 0.00174     ← 裁判本身可信
gate(official, 输入 = eg_x)   [0.02090 0.08031 1e-05 5e-05]
gate(同一公式, 输入 = 链的 moe0) [0.02089 0.08042 1e-05 5e-05]   ← 链自己算的 gate 三位吻合
chain moe0 vs eg_x (注入前的同一个量)           0.00688
reconstructed R'' (moe0 + gate·value) vs eg_y   0.00573
```

`eg_x` 与 `eg_y` 都是 dump，所以 `eg_y − eg_x` 就是增量本身，不必解 Engram embedding 的布局。
修掉这个口径后 L1 attention 从 **0.379 → 0.0192**（cos 0.999817），与用金标 `eg_y` 直接反解的
0.0218 一致。**结论：链的 Engram 注入（gate 逐 stream、value 共享、只写非零 stream）是对的。**

#### (b) `y_table.py`：80 个子层逐 kernel 打分

`y = (R' − combᵀR)/post`，用四流最小二乘（`y_i = Σ_j post_j z[j][i] / Σ_j post_j²`，自动给条件好的
stream 加权；L1/L14 这类 `post` 小、`combᵀR` 主导的层反解会放大，属于口径限制，已在表里标注）。
`own in` = 用链自己的输入反解（真实内核输出）；`gold in` = 用金标输入反解（只在输入还靠得住的早层有效）。

| 子层 | post | 链 vs 金标 `y` (own in) | 参考自身散布 |
|---|---|---|---|
| attn0 | 0.1362 | **0.00498** / cos 0.999988 | 0.00000 |
| attn1 | 0.1917 | **0.01919** / cos 0.999817 | 0.00000 |
| moe1 | 0.7243 | 0.03557 / cos 0.999379 | 0.00000 |
| attn2 | 0.2194 | 0.04041 / cos 0.999185 | 0.01291 |
| moe2 | 0.7965 | 0.04440 / cos 0.999021 | 0.02467 |
| moe7 | 0.8831 | 0.21185 / cos 0.978668 | 0.05960 |
| moe16 | 1.9642 | 0.22773 / cos 0.974496 | 0.21561 |
| attn39 | 1.6561 | 0.05990 / cos 0.998212 | 0.07196 |
| **全部 80 个 mean / median / worst** | | **0.0971 / 0.0936 / 0.2277** | **0.1091 / 0.1060 / 0.3365** |

**链的中位数（9.4%）低于参考自己四次启动的中位数（10.6%）；77 个可比子层里有 50 个链比"参考的另一次
启动"更接近参考。** 复现：`python y_table.py --layers 0-39`。

#### (c) L1 那 0.92% 的归因（不再用"参考有噪声"搪塞）

参考在 L0/L1 是逐 bit 确定的（§12.9 更正），所以 0.92% 是真的。但它完全由 L0 的舍入地板传播而来：

1. L0 残差 = 0.26%，而 bf16 的相对步长是 2⁻⁹ = 0.195% ⇒ **0.26% ≈ 1.3 ulp**，已在表示精度地板上；
2. 链的 `moe0`（= L1 attention 的输入 `eg_x`）比金标差 0.69%；
3. 反解出的 attention `y` 从 `R''` 到 `y` 的放大约 3.4× ⇒ 0.69% → 1.9%（`y` 级），残差级 0.92%；
4. 我自己独立写的 fp64 参考（`attn64.py`，权重+KV 池全 fp64）对同一份金标 `y` 是 **L0 1.82% / L1 2.19%**，
   比链更远。也就是说**目前没有任何实现比链更接近 SGLang**，L0/L1 的差已到"实现之间"的极限。

**所以 L1 不能再压**：要压必须让链在 L0 的每个 bf16 舍入点上都与服务端一致，那不是舍入误差的
"修"，是要求位级同构。

#### (d) 专家级 MoE 复算：`moe_expert_ref.py`（cos 0.9991–0.9995）

`moe_ref.py` 对 `mpost_x` 只有 cos 0.23 的谜团解开了 —— 不是 MoE 的数学，是**权重的 scale 布局**：
checkpoint 里有两种尺度网格，routed 专家是**逐行** `[out, in/32]`（`(2304,160)`），而 shared 专家/注意力
线性层是真正的 **32×32 块** `[out/32, in/32]`（`(72,160)`）。用逐行公式去展开块状 scale 会整片错位。

`moe_expert_ref.py`（新）按 dtype 分派（int8 → fp4 e2m1 拆半字节，float8_e4m3fn → 直读）、两种 scale
布局分别展开、SwiGLU 按 `model.py Expert.forward` 的钳位顺序（up 双边、gate 只钳上界，`swiglu_limit=10`），
gate 用已在 §12.11 被内核对过的转写。结果：

| 层 | relRMS vs `L{LL}_mpost_x` | cos |
|---|---|---|
| 1 | 0.03142 | +0.9995091 |
| 2 | 0.02976 | +0.9995703 |
| 5 | 0.03058 | +0.9995346 |
| 12 | 0.03339 | +0.9994426 |
| 17 | 0.03789 | +0.9992856 |
| 26 | 0.04363 | +0.9990642 |
| 35 | 0.03747 | +0.9993214 |

这比 §12.5 的反解更强：路由是独立重算的（且与内核 `pairs.bin` 的 token 0 位级一致），专家 GEMM、
fp4 反量化、块尺度、钳位、shared 加和全部在一条链上，只剩 3–4%（fp4 本身的量化噪声量级）。

> 顺带纠正一个 dump 读法：`moe{L}_r{r}_pairs.bin` 是 `(weight_f32, expert_id_i32)` **交错**，
> 而且这份运行里**最后一个 token 的槽没写**（6 个 0.25 的假平局）。要对比就得用真实 token 的槽，
> 且金标的 `_x` 行对应的是链的 **token 0**，不是最后一个 token。

#### (e) 新跑一遍 40 层：80/80 子层 + logits 逐 bit 相同

`/tmp/chain_fresh`（全新 `full_decode.py`，40 层，2.5 min build + 4.1 ms/step）：

```
streams_out  80/80 sublayers bit-identical to /tmp/chain40
logits      identical (n=129280, max|d| = 0.000e+00)
```

所以链是确定性的，下面这张表用新 dump 复算，与用 `/tmp/chain40` 的结果**完全一致**：

```
  sub     chain-A          chain-B          A-B             (relRMS / cos)
  attn0    0.0026/+1.00000 0.0026/+1.00000 0.0000/+1.00000
  attn1    0.0092/+0.99996 0.0092/+0.99996 0.0000/+1.00000
  moe1     0.0139/+0.99990 0.0139/+0.99990 0.0000/+1.00000
  attn2    0.0206/+0.99979 0.0206/+0.99979 0.0063/+0.99998
  moe2     0.0241/+0.99971 0.0236/+0.99972 0.0115/+0.99993
  ...
  moe7     0.0381/+0.99928 0.0868/+0.99626 0.0880/+0.99619   ← 第一次路由平局
  ...
  attn39   0.1140/+0.99348 0.0884/+0.99615 0.1076/+0.99421

  chain vs SGLang A (dec_coef2)  mean 0.0814  median 0.0933  worst 0.1471
  chain vs SGLang B (dec_coef8)  mean 0.0751  median 0.0801  worst 0.1331
  SGLang vs itself (A-B)         mean 0.0792  median 0.0838  worst 0.1353
```

**关键读法**：全程 80 个子层上，链对 B 的平均距离（0.0751）**小于**参考 A 对 B 自己的距离（0.0792）。
在 L7 那个平局上，A 与 B 彼此差 8.8%，链离 A 3.8%、离 B 8.7% —— 链落在参考自身的两个分支之间，
而不是另一个偏离带。这就是"落进参考散布带内"的定量版本。

---

### 12.13 参考本身从 L2 起就不确定（同输入、同系数差 1.1%），所以「每层 <1%」物理上不可达

这是本轮最重要的一条，它把验收标准从"能不能再压"变成了"极限在哪"。

`dec_coef2 / 6 / 7 / 8` 是四次独立启动（写盘时间 13:24 / 02:13 / 02:55 / 03:26），却在 **L0、L1 的每个量上都逐 bit
相同**。如果它们是不确定的，L0 就该有差；如果是确定的，L2 之后就不该有差。逐量定位（`dec_coef6` vs `dec_coef7`，层 2）：

```
L02_in        BIT-IDENTICAL       ← 输入完全相同
L02_ct_pre    BIT-IDENTICAL       ← mHC 系数完全相同
L02_ct_post   BIT-IDENTICAL
L02_ct_comb   BIT-IDENTICAL
L02_apost         0.004925        ← attention 输出不同
L02_apost_x       0.011016        ← attention 自己算的 y，差 1.10%
L02_mpost     BIT-IDENTICAL       ← 同一个输入的 FFN 输出却完全相同
L02_mpost_x   BIT-IDENTICAL
```

**同样的输入、同样的系数、同样的 FFN 输出，attention 的输出不同。** 这就是 SGLang 自己的 attention kernel
从 L2 起不确定的直接证据（不是 dump 竞争）：

- 差异 **5.64 个 bf16 ulp，分布在 84.1% 的维度上**（不是个别元素的坏值）；
- 两边对同一重建公式的残差几乎一模一样（0.294757 vs 0.294772），也就是两侧的数据完整程度相同；
- 全 40 层逐子层量化后，散布随深度单调放大（下表 `A-B` 列）。

所以 requirement 里「每层 relRMS well under 1%」的**参考侧下限**是：

| 层 | 参考自身（6 vs 7） | 层 | 参考自身 |
|---|---|---|---|
| 0 / 1 | **0.000000** | 8 | 0.085 (y 级 0.061) |
| 2 | 0.0057（y 级 0.011） | 17 | 0.127 (y 级 0.149) |
| 3 | 0.0160 (0.026) | 25 | 0.090 (0.107) |
| 7 | 0.0336 (0.047) | 39 | 0.109 (0.072) |

L2 起参考已经超过 1%，L8 之后超过 5%。**任何实现都不可能比参考与自己的距离更近**，所以那个门槛在 L2 之后
不存在可达的判据。可达的判据是「链落在参考自身的散布带内」——见 §12.12(e)，链对 B 的 mean 0.0751 < 参考
A 对 B 的 0.0792。

### 12.14 teacher-forced：attention 内核在 40 层上的逐层判决（全部在参考地板处）

有 `DSV41_TEACH=1` 时每个子层**读取 SGLang 自己的残差**，因此比较的是"这一个内核"，不含上层累积漂移。这正是
goal 里那句话的最细粒度版本（`attn{L} reads mpost(L)`、`moe{L} reads apost(L)`）。`{sub}_teach_out.f32`
是 launch 结束后立刻抓的 fp32 行 0，是真实写出值（不是注入的输入：attn0 对 `L00_apost` 是 0.00265，
如果抓到的是注入值会是 ~0.5）。判据表 `teacher_table.py`：

```
sub       vs golden         cos  SGLang-vs-itself    verdict
attn0       0.00265   +0.999996 0 (deterministic) above floor
attn1       0.00776   +0.999970 0 (deterministic) above floor
attn2       0.01639   +0.999866           0.00569 above floor
attn3       0.01664   +0.999862           0.01570 AT/BELOW FLOOR
attn4       0.02792   +0.999610           0.02069 AT/BELOW FLOOR
attn5       0.01691   +0.999857           0.02406 AT/BELOW FLOOR
attn6       0.01203   +0.999928           0.02704 AT/BELOW FLOOR
attn7       0.02318   +0.999731           0.03193 AT/BELOW FLOOR
attn8       0.03478   +0.999397           0.03285 AT/BELOW FLOOR
attn9       0.00839   +0.999965           0.03436 AT/BELOW FLOOR
attn10      0.01318   +0.999913           0.04005 AT/BELOW FLOOR
...
attn14      0.04023   +0.999192           0.05119 AT/BELOW FLOOR
```

**attention 每一个子层都在 0.003–0.04，cos > 0.999，而且从 L3 起全部低于参考自己的 6-vs-7 距离。** 复现：
`DSV41_TEACH=1 full_decode.py ... --dump-dir /tmp/chain_teach && python teacher_table.py`。

> **这条表里的 MoE 行不可用**（0.24–0.60），那是工具缺陷不是内核缺陷，已在 §12.15 定位并给了独立判据。

### 12.15 MoE 内核 == checkpoint 数学（相同输入、相同路由，40 层）

teacher 模式对 MoE 会额外用 host 侧 `apply_pre` 覆盖 `x`（`full_decode.py:640`），而这个 `x` 本应由上一层的
`hc_post_combine` 从金标残差写出。实测这个覆盖让 MoE 看起来差 0.24–0.60，但排查后确认**不是内核问题**：

- 路由完全没变（用两种 x 各自算 gate，top-6 集合在 moe1/2/5/12/17 上全部相同）；
- 加 `DSV41_TEACH_KERNEL_X=1`（不覆盖 x，让内核自己从金标残差算）仍然是 0.32 —— 说明 teacher 模式对 MoE 还有
  别的接线问题，该路径**不用于判定 MoE**；
- 改用不经过 teacher 模式的判据（`moe_expert_ref.py --from-chain`）：取链自己的 `x` 和它 dump 的路由，用
  checkpoint 权重（fp4 e2m1 + 两种 scale 布局）算"数学上应该得到的 y"，再与从链 `streams_out` 反解出的**内核
  自己的 y** 比：

| 层 | checkpoint 数学 vs 内核 y | cos |
|---|---|---|
| 1 | 0.03256 | +0.9994697 |
| 2 | 0.02691 | +0.9996387 |
| 5 | 0.03283 | +0.9994696 |
| 12 | 0.03386 | +0.9994406 |
| 17 | 0.03758 | +0.9993185 |

**跑满 40 层（`moe0`–`moe39`，40/40 全部有值）：mean 0.03642，min 0.01896（L39），max 0.04620（L27），
cos 全部 ≥ 0.9989496。** 也就是每一个 MoE 内核都在它自己输入下复现了 checkpoint 的数学，没有一层的
异常值能说明"某个专家算错了"。

而且内核 dump 的路由（`moe{L}_r0_pairs.bin` 的 token 0 槽）与 checkpoint gate 的 top-6 与权重**逐位一致**。
所以 MoE 内核就是 checkpoint 的数学，剩下的 2.7–3.8% 是 **fp4 量化本身的量级**（专家权重是 int8 打包的 e2m1）。
**⚠ 这句结论已作废：那 2.7–3.8% 其实是我们的参考漏了一次 `bf16r(silu(g)·u)`，见 §12.16 —— 补上后 40/40 层 ≥0.9999。**

> 顺带记一个 dump 读法坑：`moe{L}_r{r}_pairs.bin` 是 `(weight_f32, expert_id_i32)` **交错**，而且这份运行里
> 最后一个 token 的槽没写（6 个 0.25 的假平局）。金标 `_x` 行对应链的 **token 0**，不是最后一个 token。

### 12.16 更正 §12.15 的结论：那 2.7–3.8% 不是 fp4 量化，是我漏了一次 bf16 舍入 —— MoE 现在是**精确**的（40/40 四个 9）

§12.15 把剩余的 3.6% 归给"fp4 量化本身的量级"，**这是错的**。fp4 权重在内核和参考里是同一批数值（不是我重构的
近似），所以它不可能成为两者之差。真正的漏项在激活侧：

```c
// tasks/act_quant.cuh:348  act_quant_swiglu_task
y[k] = bf16r(g / (1.0f + expf(-g)) * u);
```

SwiGLU 的乘积在进 fp8 量化**之前**还要过一次 bf16。参考少了这一步：每个元素偏 0.2%（bf16 半 ulp），而 fp8
步长是 3–6%，于是约 13% 的元素会跨过量化边界，累积在每个专家上是 1.3–3.3%。补上后**逐专家 relRMS 直接归零**：

| 专家（L1） | 11 | 15 | 86 | 100 | 178 | 300 |
|---|---|---|---|---|---|---|
| 补 bf16r(h) 前 | 0.01692 | 0.01767 | 0.03301 | 0.01330 | 0.01434 | 0.01385 |
| 补 bf16r(h) 后 | **0.00000** | **0.00000** | **0.00000** | **0.00000** | **0.00000** | **0.00000** |

跑满 40 层（`moe_table.py --quant`）：

| | relRMS mean | median | min | max | cos mean | cos min | **≥0.9999** |
|---|---|---|---|---|---|---|---|
| 修正前 | 0.01530 | 0.01476 | 0.00326 | 0.02480 | +0.9998771 | +0.9997205 | 18/40 |
| **修正后** | **0.00461** | 0.00398 | 0.00229 | 0.01143 | **+0.9999879** | **+0.9999347** | **40/40** |

而且剩下的 0.4% 已经是**仪器下限**，不是内核误差：把参考自己的 y 合成 `R' = post·y + combᵀR`、round 到 bf16
（残差 buffer 的存储精度）、再反解回来，得到的误差与"内核 y vs 参考"同量级：

| 层 | 0 | 1 | 8 | 20 | 30 | 39 |
|---|---|---|---|---|---|---|
| 内核 y vs 参考 | 0.00277 | 0.00376 | 0.00530 | 0.00427 | 0.00730 | 0.00229 |
| 仪器下限（已知 y 往返） | 0.00200 | 0.00297 | 0.00478 | 0.00366 | 0.00693 | 0.00124 |

**结论：MoE 内核在它自己的输入下逐位复现 checkpoint 数学，误差已经低于我们能测量的分辨率。**

这条也把残差的归属彻底改写了：之前认为是"fp4 噪声"的那 1.5%，实际来自我们自己的参考。**链 vs SGLang 的
剩余分歧全部在 attention 的算子顺序和路由翻转上**，与 MoE 算术无关（路由翻转见 §12.10，参考自身地板见 §12.13）。

配套脚本：`stage_probe.py`（分阶段逐字节判定：stage1 `act_quant` 逐字节一致、stage2 `topk_route` 集合+权重一致、
stage3/4 因 scratch 被清零无法读）、`quant_probe.py`（两级量化的净影响：3.28% → 1.95%）、`variants.py`（实现变体
扫描：`quantize(w·h)` 更差、`fp8 h` 优于 `bf16 h`/`fp32 h`）、`moe_table.py`（40 层表）。

> 两个 dump 读法坑（这次踩到）：① MoE 的 `routed_rows`/`sd_part`/`h_q` 是 **TP=8 的每卡 partial**，必须把
> `moe{L}_r0..r7` 八个 rank **相加**才是 y 的对应量，只看 rank 0 会得到 1/8 量级、cos 0.35 的假象（`sum_store.cuh`
> 开头写了这个语义）；② `routed_rows` 的行序是**专家 id 升序**，不是 gate 的输出顺序。

### 12.17 运算级逐 bit：MoE 内核**逐 bit 等于** checkpoint 数学（cos 恰好 1.0），剩下的唯一 1-ulp 环节是 gate 点积

问的是"能不能到 1.0"。分两层，答案相反：

- **对数学规格**：能，而且已经做到了 —— 本节。
- **对 SGLang 可执行文件**：不能 —— 它自己两次启动从 L1 起就不同（L1 1.46 ulp / 27.7% 位不同，L2 4.02 ulp /
  56.8%，L17 30.4 ulp）；天花板 L1 0.9999838 → L39 0.9950254。参照物自己在漂，1.0 不是有定义的目标。

#### 为什么这件事可做（两个前提）

1. **每个乘积在 fp32 里都是精确的**：fp4 e2m1 只有 1 位尾数、fp8 e4m3 有 3 位，乘积至多 5 位有效位；scale 又是
   2 的幂。所以唯一的舍入在求和与收尾，不在乘法。
2. **路由权重必须用内核自己的 fp32 值**。这是关键：我算的权重与内核 `pairs` 里的差 **1 ulp**（6 个专家里 5 个），
   而这个 1 ulp 会乘满整行 —— 表现正是"89% 元素差 ±1–2 ulp、正负对称、均值 0"。用内核的值之后：

| 口径 | 逐 bit 相同 | max\|d\| |
|---|---|---|
| 我算的权重（差 1 ulp） | 573 / 5120 | 1.86e-09 |
| **内核 pairs 的权重** | **5120 / 5120** | **0.000e+00** |

#### 全量核对（L1，6 专家 × 8 rank）

TP 切分是**切 K**：rank r 拥有 `[288r, 288(r+1))`（288 = 2304/8，用逐 32 列块归因确定）。每格 5120 维：

| expert | rank0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 11 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 |
| 15 | 5120 | 5120 | 5120 | 5120 | **5100** | **5098** | 5120 | 5120 |
| 86 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | **5117** |
| 100 | 5120 | **5118** | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 |
| 178 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | **5112** |
| 300 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | 5120 | **5117** |

**42/48 个 (expert, rank) 单元格 100% 逐 bit 相同，全部 48 格 max\|d\| = 7.45e-09。** 没全同的 6 格各差 2–22 个
（共 5120）元素、每个差 1 fp32 ulp —— 正好落在"288 项 fp32 求和**不能精确表示**"的位置，那里累加顺序才起作用。

shared expert（无路由权重，纯算术检验）：**relRMS 7.225e-08，cos = 1.0000000000**（float64 下恰好 1），
max\|d\| 2.14e-08。

#### 唯一剩下的 1-ulp 环节：gate 的 5120 项 bf16 点积

路由权重来自 `gemm_tile`（`tcgen05.mma kind::f16`，bf16×bf16 + fp32 累加）。bf16×bf16 的乘积有 16 位尾数，
**求和不再精确**，所以最后 1 ulp 由累加顺序决定。我扫了 11 种顺序（逐元素/分块 4–256/pairwise/逆序），
最多只能复现 **3/6** 个权重；torch 的 `bf16 @ bf16` 更差（它返回 bf16，logits 差 4.6e-6）。这是专有累加器，
和 §12.16 里 w2 的 1 ulp 同一个性质。

#### 结论

- **运算级**：MoE 的 `act_quant` → w13 → bf16 → 钳位 → SwiGLU → bf16 → `act_quant` → w2（mxfp4 tcgen05 块缩放）
  → 乘路由权重 → TP K 切分求和，**全部逐 bit 等于 checkpoint 数学**；cos 就是 1.0。
- §12.16 说的"剩下 0.4% 是仪器地板"要修正为：**一半是 bf16 残差往返的仪器地板，另一半是我路由权重的 1 ulp**。
  两者都不是内核算术。
- 想在这条链上再往前，只剩 gate 点积那一环，而那需要复现 `tcgen05.mma kind::f16` 的内部累加 —— 换个说法：
  **除了一个 5120 项点积的最后 1 位，MoE 已经是 1.0。**

脚本：`bitexact.py --layer 1 --dump /tmp/chain_q`（上表与结论直接可复现）。

---

### 12.18 attention 侧同样对完了：三个量化节点逐字节相同、五个 GEMM 全在 fp32 地板，只剩 `o` 的 bf16 模糊度

把 §12.17 的方法搬到 attention 上：`full_decode.py --dump-all attn0,attn1` 把每个节点的缓冲区都写盘，
`attn_nodes.py` 直接读内核写的 buffer（不是从 bf16 残差反解），按 checkpoint 的数学逐节点核对。

#### 每个节点的字节布局（先解出来，否则会误判）

| 节点 | 字节数 | 含义 |
|---|---|---|
| `x_q` | `zq_bytes(8, 5120) = 61440` | e4m3 `[8][5120]` + 每 128 列一个 512 B 的 scale 块 |
| `wq_a_part` | `8·1280·4·4 = 163840` | fp32 `[P=4][T=8][K=1280]` 的 K 部分和 |
| `qr_q` | `zq_bytes(8, 1280) = 15360` | 同上，e4m3 |
| `wq_b_part` | `8·4096·4 = 131072` | 本 rank 8 个 head，1 个 part |
| `wkv_part` | `8·512·4·2 = 32768` | 2 个 K part |
| `o_q` | `8·4096·2 = 65536` | **bf16**（`BF16O`：wo_a 是 bf16，o 不做 act_quant） |
| `wo_a_part` / `oa_q` / `wo_b_part` | `65536` / `12288` / `327680` | 2 part / e4m3 / 2 part |

`SF_CHUNK = 512` 覆盖 **32 个 token 槽 × 128 列**（4 个 32 列块），布局是
`chunk[列/128][(t%32)·16 + 4·(t/32) + 块号]` —— 不是每行一份 scale，按 8 token 密排会全错。

#### 三个坑（都会让数字看起来"差 6%"而不是"差 1 个舍入"）

1. **e4m3 取整不能用 `frexp`**。`np.frexp` 把尾数归一到 `[0.5, 1)`，即 binade 高了一档；按 1/8 取整就成了
   `[32,64)` 内步长 8 而不是 4，**一半的可表示值取不到**。照这个写 `x_q` 只有 94% 位相同（relRMS 6.2e-2）。
   改用 torch 的 `float8_e4m3fn` cast（底层就是 `cvt.rn.satfinite.e4m3x2.f32`）后 **40960/40960**。
2. **`float8_e8m0fnu` 转 float 已经是乘数**。`.float()` 给出 `2^(byte-127)`，再套一次 `pow(2, s-127)` 会得到
   `2^-126` —— 所有 fp8 权重变成 ~1e-37（rms 4.8e-37），GEMM 的 cos 会变成 0.0000。
3. **`q_norm` 的输入是 bf16，不是 fp32 的部分和**。`q_norm.cuh`：`add_parts8` 按 part 序求和 → **`bf16r`**（那是
   GEMM 的 bf16 输出）→ RMSNorm；而且 `y = bf16((x·r)·gamma)` 是**两次独立舍入**，`mean(x²)` 的求和顺序也不是
   torch 的（线程 8 列 → warp 蝶形 → 5 个 warp 依次）。少了那个 `bf16r` 是 5.9e-3；把整条链照抄下来是 **0**。

同样，`act_quant.cuh:280` 在量化公式前还有一次 `bf16r`（GEMM 的 bf16 输出），少了它 `oa_q` 差 1.7e-2。

#### 逐节点结果（L1，rank 0；8 个 rank 全部同结论）

| 节点 | relRMS | cos | 逐 bit 相同 |
|---|---|---|---|
| `x_q` = act_quant(x) | **0** | 1.000000000 | **40960 / 40960** |
| `wq_a` sum = x_q·W_qaᵀ | 4.06e-08 | 1.000000000 | 9643 / 10240 |
| `qr_q` = act_quant(q_norm(wq_a)) | **0** | 1.000000000 | **10240 / 10240** |
| `wq_b` = qr_q·W_qbᵀ | 6.02e-08 | 1.000000000 | 31477 / 32768 |
| `wkv` sum = x_q·W_kvᵀ | 7.99e-08 | 1.000000000 | 3785 / 4096 |
| `wo_a` sum = o·W_oaᵀ | 3.78e-07 | 1.000000000 | 7245 / 8192 |
| `oa_q` = act_quant(wo_a) | **0** | 1.000000000 | **8192 / 8192** |
| `wo_b` sum = oa_q·W_obᵀ | 2.06e-08 | 1.000000000 | 40256 / 40960 |

**三个量化节点逐字节相同，五个 GEMM 全部 `cos` 恰好 1.000000000、worst 7.1e-7（8 个 rank 的最大值）。**
和 §12.17 的 MoE 同一个性质：e4m3 只有 4 位有效位、乘 2 的幂，乘积精确，5120 项以内求和也精确，
剩下的 1e-8 是 fp32 累加顺序。`[P][T][K]` 的 part 划分是调度细节（每个 part 覆盖哪些 K 列不固定），
语义量是它们的和 —— 所以我比较 sum，不比较 part。

#### 唯一剩下的一环：`o_q`，而它是 bf16 模糊度，不是 bug

把 `wq_b` 的输出过 rope + sink softmax 重算 `o`，和内核的 `o_q` 差 4.6e-3（L0）/ 8.5e-3（L1）＝ 1.2 / 2.2 个
bf16 ulp。我按 `inference/kernel.py sparse_attn_kernel` 的两段 64-key online softmax 精确复现（per-block max、
`sum·sc + ΣP`、`O·sc + bf16(P)·K`、最后 `sum += exp(sink-m)`），**1.00x**，没有改善 —— 说明不是分块结构。

三个判据都指向"这就是地板"：

- **三角是等边的**：`o` 我 vs 内核 0.0085；`y` 我 vs 内核 0.0204、我 vs SGLang 0.0186、内核 vs SGLang 0.0198。
  若有共模误差（比如 K 用错了），两种 `o` 会一起偏向同一个错答案，彼此很近而一起远离金标；实测彼此一样远，
  **没有共模项**，q / K / sink / scale 都是对的（scale 扫描也确认恰好 `1/√512`，不是 `1/√576`）。
- **放大倍数实测**：把内核的 `o` 扰动 ε 后走**同一条** wo_a/act_quant/wo_b，
  ε = 0.0085 → y 变 **0.0186**，正好对上实测的 0.0198。放大 ~2.2× 来自 `oa_q` 的 e4m3 分桶翻转
  （和 MoE 里 h 的 13% 元素跨 fp8 边界是同一个机制）。
- **这条链的下限很硬**：ε = 0.001（0.26 个 bf16 ulp）就已经让 y 动 0.00625，**比参考自己两次启动的 0.0057 还大**。
  所以 L1 的 y 在任何实现下都不可能比 ~4e-3 更好 —— 除非 `o` 逐 bit 相同。

要逐 bit 就得复现 tilelang 的 per-lane `reduce_sum`（每 lane 8 个值 → 2 个 key 半区 → lane^2 → lane^1）和
`mma.sync m16n8k16` 的内部累加树；内核自己就是这么做的（`sparse_attn.cuh` 的头注释逐条对应官方 kernel），
但在 numpy 里只能逼近到 2 ulp。**结论：attention 没有漏舍入**（对比 MoE 那次真找到了 `bf16r`），
L1 的 0.92% 是 `o` 的 bf16 经 e4m3 分桶放大后的地板。

脚本：`attn_nodes.py --layer {0,1} --rank {0..7}`、`attn_core.py --layer 1`。

---

### 12.19 追到逐 bit：attention 子层**逐字节等于参考实现**（跑官方 kernel 判定）

§12.18 留下唯一一个开口：`o_q` 在 numpy 里只能到 2 bf16 ulp。正确的做法不是再换一种 numpy 写法，而是
**把官方 kernel 本身跑一遍** —— `inference/kernel.py` 就是 checkpoint 自带的参考实现，SGLang 执行的就是它。
用 megakernel 自己缓冲区里的 q 和 K 喂进去，问它的 `o` 是否逐 bit 等于 megakernel 的 `o_q`。

#### `sparse_attn`：32768 / 32768

喂进去的东西（全部取自 megakernel 自己的 buffer，路径上不做任何重新推导）：

    q      = bf16( rope( bf16( Σ wq_b_part 的 K parts ) ) )            <- attn1_r{r}_wq_b_part.bin
    K_now  = act_quant( bf16( rope( rms_norm( bf16( Σ wkv_part ) ) ) ) )  <- attn1_r0_wkv_part.bin
    K_old  = 参考池里的行（megakernel 这一步只写新行，历史行两边读的是同一份字节）
    sink   = layers.1.attn.attn_sink,  idxs = arange(128)（环按官方列表序，最老在前）

| | 逐 bit 相同 |
|---|---|
| q 用 kernel 的 `fma` rope | **32768 / 32768**（8 rank × [8,512]）|
| q 用 torch 的复数 rope | **32768 / 32768**（1 ulp 的 rope 差异被 bf16 吸收）|

**`o_q` 闭合。** numpy 的 4.6e-3 / 8.5e-3 从来不是内核的误差，是 numpy 复现不了 `mma.sync` 的内部累加树 ——
而 megakernel 的 `sparse_attn.cuh` 用的**就是** `mma.sync.aligned.m16n8k16`（同一条指令），所以逐 bit 相同。

#### 五个 GEMM：fp32 **不是**逐 bit；逐 bit 的是它们的"接口值"（bf16）

先说清楚口径，因为这里容易自欺：

- **fp32 这层没有逐 bit。** `wq_a_part` 等 buffer 本身是 fp32，真实数字是 411/1280（wq_a）、2562/4096（wq_b）、
  149/512（wkv）、4226/5120（wo_b），relRMS 4e-8–1.3e-7。
- **bf16 那层不是我把检验放宽了，而是消费者真正拿到的数。** 四个节点的消费者第一件事都是把 fp32 和舍入到 bf16，
  源码里写死：

  | 节点 | 消费者 | 证据 |
  |---|---|---|
  | `wq_a_part` | `q_norm.cuh` | `x[k] = __bfloat162float(__float2bfloat16_rn(x[k])); // the GEMM's output` |
  | `wq_b_part` | `sparse_attn.cuh:230` | `x[e] = bf16r(x[e]); // wq_b's output` |
  | `wkv_part` | `kv_post.cuh:5` | `Add wkv's P K parts in part order -> bf16 (wkv's bf16 output)` |
  | `wo_b_part` | `sum_store` | `sent = nt((mpk.num_gpus * T, HID), bfloat16, "y_sent")` |

  所以 `bf16(Σ parts)` **就是**跨过边界的那一个数；fp32 那 1 ulp 在下游没有观察者。这是"接口值相同"，
  不是"内核逐 bit"。

拿官方 `fp8_gemm_kernel`（block 32、e8m0 scale、fp32 累加）跑 megakernel 自己的 `x_q`/`qr_q`/`oa_q` 字节。
统计只算 token 0（链只跑一个 token，第 1–7 行全零，否则每个数都会被虚高到 7/8）：

| 节点 | K | N | parts | fp32 逐 bit | **接口值 bf16 逐 bit（8 rank 合计）** |
|---|---|---|---|---|---|
| `wq_a` | 5120 | 1280 | 4 | 411 / 1280 | **10240 / 10240** |
| `wq_b` | 1280 | 4096 | 1 | 2562 / 4096 | **32768 / 32768** |
| `wkv` | 5120 | 512 | 2 | 149 / 512 | **4096 / 4096** |
| `wo_b` | 1024 | 5120 | 2 | 4226 / 5120 | **40959 / 40960** |

权重从来没有按 bf16 比过：`wq_a/wq_b/wkv/wo_b` 在 checkpoint 里是 e4m3 + e8m0，我按 `e4m3 × 2^k` 精确反量化到
fp64 再乘（4 位尾数乘 2 的幂，在 bf16 和 fp64 里都是精确的）。bf16 只用在 GEMM 的**输出**上，因为消费者的
第一件事就是它。

`wo_b` 那唯一一个（rank 0, n3699）是**灾难性抵消**：结果 2.521e-05，而参与求和的各项是 2.637e-02 量级，
所以"各项的 1 个 fp32 ulp"在这个结果上就是 ~2000 ulp，正好落在 bf16 两个邻居的中点一侧。无解，也不该修。

顺带一个交叉检验：把每个 part 单独和官方结果比，是 **0 / 1280** —— 说明 part 确实是 K 切片（不是重复计算），
而它们的和是 1280/1280。

#### 为什么 fp32 上不能 1.0（可命名的硬件边界）

- megakernel：`gemm_fp8_tile.cuh` 用 **`tcgen05.mma kind::mxf8f6f4 block_scale`** —— 每个 32 列的 scale
  由 MMA 指令自己应用（scale 融进累加）。
- 参考：tilelang 0.1.12 的 `T.gemm` 在 e4m3 上**只能**降到 `tl::mma_sync<e4m3,e4m3,f32,16,8,32>`
  （`instruction/mma.h`），scale 是 MMA 之后显式乘再累加。我把 `pass_configs` 清空重编译确认过：仍然是
  `mma.sync`，这个版本没有 tcgen05 的 `T.gemm` 路径。

两条路的数学定义相同，fp32 归约的树不同。而且**谁都不比谁准**：距精确值（e4m3×e4m3 乘积在 fp64 里精确、
scale 是 2 的幂、5120 项放得下 53 位，所以 fp64 求和就是正确舍入值）的平均偏差，以该张量最大元素的 1 fp32 ulp 为单位

| | wq_a | wq_b | wkv | wo_b |
|---|---|---|---|---|
| megakernel | **0.075** | 0.043 | **0.140** | **0.019** |
| 官方 tilelang | 0.157 | 0.038 | 0.176 | 0.037 |

两者都在 0.2 fp32 ulp 以内，而且在四个节点里 megakernel 有三个更近。**这不是"谁错了"，是 reassociation。**

#### 到这里 attention 子层的账

| 节点 | 判定 |
|---|---|
| `x_q` / `qr_q` / `oa_q` | **逐字节相同**（这些 buffer 本来就是 e4m3，比的就是它的真实存储精度）|
| `o_q` | **逐 bit 相同，32768/32768**（同样的道理：`BF16O` 让这个 buffer 本身就是 bf16，比的是真实存储精度，没有放宽）|
| `wq_a` / `wq_b` / `wkv` / `wo_b` | fp32 **不**相同（~1 ulp reassociation）；**接口值 bf16 逐 bit 88063 / 88064** |

这里必须区分两类：`x_q`、`qr_q`、`oa_q`、`o_q` 比的是 buffer 自己的 dtype（e4m3 / bf16），那是**严格的**逐 bit；
四个 fp8 GEMM 的 buffer 是 fp32，**没有做到严格逐 bit**，做到的是"消费者看到的那个值相同"。

为什么 bf16 那一层几乎是全同：1 个 fp32 ulp 只有 bf16 ulp 的 1/65536，要翻一次 bf16 舍入得恰好落在中点 2^-16
邻域，88064 个真实值里期望不到 1 个 —— 实测正好 1 个，且是抵消点。这是**下游不可见**，不是"内核精确"。

脚本：`tilelang_check.py --layer 1`（sparse_attn）、`tilelang_gemm.py --layer 1 --rank {0..7}`（fp8 GEMM）。

---

### 12.20 按 checkpoint 自身的 dtype 逐类判决：fp8 ✓ / bf16 ✓ / **fp32 做不到**（附证明）

判据按用户定的规矩来：**checkpoint 里这个张量是什么 dtype，就要求在那个 dtype 上逐 bit 相同**，不许换口径。

#### 先看 checkpoint 里到底有哪些 dtype（`layers.1.attn.*` 实测）

| tensor | dtype |
|---|---|
| `wq_a / wq_b / wkv / wo_b .weight` | **float8_e4m3fn** |
| 同上 `.scale` | **float8_e8m0fnu** |
| `attn_sink` | **float32** |
| `q_norm / kv_norm .weight` | bfloat16 |
| `wo_a.weight` | checkpoint 里是 e4m3，但 `convert.py:157-168` 加载时**无损**反量化成 bf16 |

关键观察：**checkpoint 里没有任何一个 fp32 的权重**。唯一的 fp32 张量是 `attn_sink`，而它是**原样加载**的
（两个实现读同一份 fp32 字节，中间没有任何 dtype 转换，不存在重新计算），消费公式也相同（`exp(sink - m)`
进入 softmax 分母）—— 这一点由 `o_q` 的 **32768/32768** 逐 bit 相同一并确认。
**所以 fp32 这一类，按 checkpoint 的定义，已经是逐 bit 相同的。**

#### 三类判决

| dtype 类 | 节点 | 结果 | 备注 |
|---|---|---|---|
| **fp8** | `x_q` / `qr_q` / `oa_q` | **逐字节相同** | buffer 本身就是 e4m3，比的就是真实存储精度 |
| **fp8** | `wq_a/wq_b/wkv/wo_b` 的权重与激活字节 | **逐字节相同** | 见 §12.18 |
| **bf16** | `o_q` | **32768/32768 逐 bit 相同** | buffer 是 bf16（`BF16O`），比的是真实存储精度 |
| **bf16** | `wo_a`（反量化后）/ norms | 逐 bit 相同 | e4m3×2^k 在 bf16 里精确，无反量化误差 |
| **fp32** | `attn_sink` | **逐 bit 相同**（经 `o_q` 反证）| checkpoint 里唯一的 fp32 张量 |
| fp32 | *内部* K-part 累加器（checkpoint 里不存在）| 411/1280 等 | 见下 |

#### 那个 fp32 buffer 为什么做不到 —— 这不是"我放宽口径"，是物理

`wq_a_part` 这类 buffer 是 megakernel 自己的内部布局（K 切成 P 份并行），**参考里没有对应张量**：官方
`fp8_gemm` 直接返回 bf16，从不materialize fp32 的 K 分片。所以它不是一个"checkpoint 里 fp32 的权重"。

那它和参考的 fp32 结果为什么差 ~1 ulp？`gemm_reduction.py` 把归约拆开做了 12 种软件建模
（32 项点积取 `exact` / `seq` / `tree` / `block-FP 截断` / `block-FP 舍入`，外层累加取 单累加器 / P 累加器 /
交错 P 累加器）：

| 模型 | vs megakernel | vs 参考 |
|---|---|---|
| 32-dot exact + 单个 fp32 累加器 | 411/1280 | **1253/1280** |
| 32-dot exact + 4 个 K-part 累加器 | 610/1280 | 489/1280 |
| 32-dot trunc + 单个累加器 | 413/1280 | 1238/1280 |
| ...（共 12 种） | 最高 683/1280 | 最高 1253/1280 |
| 正确舍入值（fp64）本身 | 683/1280 | 460/1280 |

两个结论：

1. **参考是"32 项精确点积 + 逐块 fp32 顺序累加"，98% 可复现**（1253/1280）。剩下 2% 是 `mma.sync` 内部细节。
2. **megakernel 用任何 IEEE 模型都复现不了**，最好只有 683/1280 —— 而那恰好就是"正确舍入值"本身。
   这说明它的误差不是任何"加法顺序"能产生的，只能来自 `tcgen05.mma kind::mxf8f6f4 block_scale` 的
   **数据通路内部**（Block-FP 对齐/截断一类）。

顺带：scale 是 e8m0（2 的幂），所以缩放放在 MMA 里还是放 epilogue、先乘 sa 还是先乘 sb，**都是精确的**，
不可能是差异来源 —— 差异只能在求和。

还有一个反直觉但重要的点：**megakernel 更准**。与正确舍入值的一致率 683/1280（53%）
> 参考的 460/1280（36%）。两者都是 ~1 fp32 ulp 的独立噪声，所以互相只有 32% 相同。
**这不是"谁错了"，是两颗不同的张量核在各自舍入。**

#### 所以要 fp32 逐 bit，唯一的路

必须让 megakernel 放弃 tcgen05 + K 分片，改用和参考一模一样的 `mma.sync m16n8k32` + 单个累加器按 K 升序累加。
代价：这四个 GEMM 失去 P 路并行（wq_a P=4、wkv P=2、wo_b P=2、wq_b P=1）。这不是 bug 修复，是**用并行度换
bit-exact**的选择题。

脚本：`gemm_reduction.py --layer 1 --node {wq_a,wq_b,wkv}`。

---

### 12.21 40 层端到端现状（2026-11-27 收盘）与 rope 表核对

#### 结论一句话

链**落在参考自己的重现散布带内**：80 个子层的 y 级平均 0.0971，低于 SGLang 两次启动之间的 0.1091；
40 层残差对 dec_coef8 平均 0.0751，而 dec_coef7 vs dec_coef8 是 0.0735。**没有系统性的算法/权重错误。**

#### 三层证据

**1. y 级（80 个子层，`y_table.py`）**

| | 均值 | 中位 | 最差 |
|---|---|---|---|
| 链 vs golden y（own input） | **0.09710** | 0.09364 | 0.22773 |
| 参考自身散布（dec_coef 2/6/7/8 两两最大） | 0.10913 | 0.10602 | 0.33649 |

**链比参考自己的两次启动更接近**：50/77 个子层上，链比 SGLang 的另一次启动更靠近 golden。

**2. 残差级（40 层 × 2 子层，`align3.py`）**

| 子层 | 链-A | 链-B | A-B |
|---|---|---|---|
| attn0 | 0.0026 | 0.0026 | 0.0000 |
| attn1 | 0.0092 | 0.0092 | 0.0000 |
| attn2 | 0.0206 | 0.0206 | 0.0057 |
| attn5 | 0.0285 | 0.0295 | 0.0241 |
| attn10 | 0.0785 | 0.0769 | 0.0400 |
| moe20 | 0.1346 | 0.1094 | 0.1091 |
| moe30 | 0.0912 | 0.0808 | 0.0890 |
| attn39 | 0.1043 | 0.0884 | 0.0997 |
| **80 子层均值** | 0.0888 | **0.0751** | **0.0735** |

链对 B 已经贴住噪声底（0.0751 vs 0.0735，差 2%）；对 A 高 21%。注意这不是度量距离，relRMS 不满足三角不等式，
所以不能读成"链在 A、B 之间"。逐子层看，链-B < A-B 有 41/77（近似抛硬币），链-A < A-B 是 0/77 —— 链系统性地
离 B 更近，离 A 更远。

**3. logits（对 golden step-1，行对齐比 `topk[1]`）**

| | 结果 |
|---|---|
| top-1 | **343，与 golden step-1 的 top-1 一致** |
| top-32 交集 | 27/32 |
| \|Δlogprob\|（golden top-32） | mean **0.385** / median 0.312 / max 1.241 |
| 扣掉一个整体偏移后 | mean 0.364（偏移不改分布形状，说明不是整体标定问题）|

一个必须记住的口径细节：**两边的分布都落在粗格子上**。链的 logits 值是 bf16 网格（20.75、23.125 都是
[16,32) 的 1/16 步长），golden 的 logprob 差分恰好是 0.125 的整数倍。32 个 token 的偏移量
（链 − golden）全部落在 `0.0085 + k/8` 上，$k$ 的分布是 −10…+6。**所以每 token 的 ±0.125 是这个存储精度的
一格，不是连续漂移**；27/32 的重合和 0.385 的平均偏差，大半是这个格子 + 尾部排名受 10% 残差影响的结果。

**4. 确定性**：`/tmp/chain40`、`/tmp/chain40b`、`/tmp/chain_fresh` 三次独立运行，**80/80 个 `x_out`、
80/80 个 `streams_out` 逐字节相同，logits 的 md5 完全相同**（`a812ee1e…`）。链本身是确定性的，上面的差别
全部来自参考侧。

#### 顺手关掉一个悬案：压缩层的 YaRN rope 表

`compress_ratios = [0,0,2×18,1×20,0,0,0]`，即 L2–L19 用 ratio 2、L20–L39 用 ratio 1，都是压缩层。
把 `weights.py freqs_of` 和 `inference/model.py precompute_freqs_cis` 的表逐元素比：

| 层 | compress_ratio | rope | 逐 bit |
|---|---|---|---|
| L0 / L1 | 0 | plain Θ=10000 | **2097152/2097152** |
| L2 / L5 | 2 | YaRN Θ=160000 factor 16 | **2097152/2097152** |
| L20 / L39 | 1 | YaRN Θ=160000 factor 16 | **2097152/2097152** |

**rope 表不是 L2+ 偏差的来源。** L2 的 0.0206（参考自身 0.0057）更可能是 indexer 的 top-k 块选择这种
**离散决策**被微小分数差翻转 —— 与 MoE gate 的 top-6/top-7 平局同一类，不是数值 bug。

---

### 12.22 「top1 32/32」的硬上限：参考自己只到 6/32，而且分叉点是一个**精确平局**

目标若读作「32 步贪心里每步 top-1 都对」，这个目标**物理上不可达**，而且不是本实现的问题：

```
两次 stock SGLang（同一 prompt、同一 8 卡、同一代码、deterministic=False）
step 0..5  一致：4588 343 6333 294 270 4245
step 6     分叉：run1 -> 344    run2 -> 3445
step 7+    上下文已分叉，两串互不相关（run2 还退化成 8-token 循环）
=> 参考对自己的 top-1 命中 = 6/32
```

分叉点是一个**精确平局**，不是误差：

| | run1 | run2 |
|---|---|---|
| top-1 | 3445 @ **-1.5747** | 3445 @ -1.3242 |
| top-2 | 344 @ **-1.5747** | 682 @ -1.8242 |
| top-1 − top-2 | **+0.0000** | +0.5000 |

run1 的两个候选 logprob **完全相同**（都是 -1.5747），argmax 只能由存储顺序决定。任何实现（包括参考
自己）在这一步都只能**抛硬币**。所以「32/32」的严格版本要求参考先和自己一致，而它不一致。

**这个指标的饱和点也一并量了**（同一对 reference 运行，逐 step 的 top-32 交集）：

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
|---|---|---|---|---|---|---|---|---|
| 交集 | **32/32** | **30/32** | 29/32 | 29/32 | 31/32 | 30/32 | 30/32 | 8→0（已分叉） |

即：`32/32` 只在 step 0 出现过，step 1 的天花板是 **30/32**。链现在 27/32。

**链离地板有多远**（step 1，26 个三方共同候选上，同一批 token）：

| | mean \|Δlogprob\| | median |
|---|---|---|
| SGLang run1 vs run2（地板） | 0.1892 | 0.1582 |
| 链 vs run1 | **0.3647** | **0.3125** |

**1.93×**，是普遍偏大（median 也 2×）而不是几个离群点，26 个 token 里 9 个已经在地板或更好。
换算成 bf16 ulp（该量级 ulp = 0.125）：地板 ≤3 ulp，链最多 9 ulp。

**5 个缺失的归类**（这条最关键，说明大半不是精度问题）：golden step-1 的 top-32 只有
**17 个不同的 logprob 值 / 9 组并列**（有一组 4 个）；链自己的 top-32 也有 **34 个** token
达到或超过它的第 32 名 → 集合本身就有 2 个 token 的模糊。5 个缺失里
3 个落在 ±2 ulp 的并列带内（-0.9 / -1.9 / -1.9 ulp），真正离群的是
**50359 (-9.9 ulp)** 和 **11756 (-3.9 ulp)**；反向多出来的 5 个里 `6575` 我们给 -4.7115
（我们排 ~12 名）而参考根本没进 top-32 —— 这条是最大的单点差异。

**head 已排除**（不是它的锅）：
1. 129280/129280 个 logits 都是**精确的 bf16 值** —— 与参考同一量化（参考候选间 gap 也是 0.125 的整数倍），
   不是我这边多舍入了一次；
2. 用链自己的 `moe39_streams_out`（4 流）+ 本层 carried pre-mix 独立反算 logits，与链自己写的
   logits 的 `max|d| = 0.072` ≈ 半个 bf16 ulp（就是输出那次舍入）。口径为
   `head(norm(pre_mix · streams))`，`pre_mix` 取 `hc_coef` 前 HC 个 fp32。

**结论**：`top1 32/32` 不可达；`top-32 交集` 的天花板是 **30/32**（参考对自己的复现率），现在是 27/32，
其中 ~3 个是并列带内的顺序问题。要把 27 推到 30，等价于把端到端误差**减半**到参考自身噪声以下 ——
而 fp8 GEMM 的 `tcgen05` vs `mma.sync` 归约树（§12.20）使逐 bit 复现不可得，所以这条路的终点
是「与参考自身的散布同量级」，不是逐 bit 相同。

**三方表（不是两方）**：链对着 A 是 27/32、对着 B 是 **28/32**、落在 A∪B（34 个 token）内是 **29/32**。
只有 `260 / 334 / 14676` 三个 token 两个参考都没选，且都在 ±3 ulp 的切点带内。A 独有的
`2552 / 50359`、B 独有的 `6575 / 8049` 说明**参考自己在这条尾巴上也不稳**（例如 `566`：A -5.4703 /
B -5.6872 / 链 -5.7115 —— 链更靠 B，`11756` 同理）。所以「27/32」里含有一部分是「拿单次参考当标尺」的偏差。

### 12.23 L2 的 indexer top-k 不是歧义源（17 个并列正好 17 个名额）

> **⚠ 本节结论已被 §12.35 推翻。** 下面那次统计把 `scores` 尾部的 **175 个填充槽（值 0.0）**算进了前 512 名——
> 0 比所有真实分数都大，于是它们挤占了名额。内核自己有 `n = min((p+1)/R, CAP)`（`indexer_topk.cuh:201`），
> 只在前 15953 个里选。**排除填充后是 506 严格更高 + 24 个精确并列、只剩 6 个空位 → 18 个槽的取舍由实现定义。**
> 结论应改为：L2 的 top-512 确实有 3.5% 的实现定义歧义（见 §12.35 末节）。

此前把 L2 的 3.6× 归给 indexer 的离散块选择。**这个假设不成立**：

`attn2` 的 `scores` 缓冲（bf16，[8, 16128]）在 512 名的切点上：**495 个严格更高、17 个精确并列于
-1.3125、正好 17 个空位** → 17 个并列候选**全部被选中**，选择集与 tie-break 顺序无关（按索引升序与
降序两种极端顺序得到的选择集**差 0 个**）。

而且官方实现里 `index_score` 也是 bf16：`fp4_act_quant(..., inplace=True)` 是**先量化再反量化回原
dtype**（`kernel.py:184`），所以 `q`/`index_k` 仍是 bf16，`einsum` 与 `sum(dim=2)` 都在 bf16 ——
与我们的 bf16 `scores` 同精度、同一批并列。**双方都是 bf16，并列是共有的，且这里名额恰好够。**

所以 L2 的 0.0206 **不是**离散翻转，而是从 L0 的种子逐层放大的连续量；`chain-A = chain-B = 0.0206`
（对两套参考完全相同）也印证它是一个固定的系统性偏差，不是随参考选择的抖动。

### 12.24 逐 stream 看 mHC：每层只有一个 stream 偏差大，合成量却在 fp64 参考的距离上

把 20480 维残差拆成 4 个 mHC stream 逐层打分（对 B 的 dump）：

| 子层 | 逐 bit 相同的 stream（%） | 每 stream 相对误差 | 合成 relRMS |
|---|---|---|---|
| attn0 | 98.3 / 98.7 / **100.0** / 47.4 | 2.9e-4, 3.2e-4, **0**, 4.4e-3 | 0.0026 |
| attn1 | 11.1 / 44.5 / 66.3 / 30.6 | **1.6e-2**, 3.8e-3, 1.9e-3, 5.6e-3 | 0.0092 |
| attn2 | 5.8 / 19.5 / 65.4 / 25.1 | **3.0e-2**, 8.9e-3, 2.0e-3, 7.0e-3 | 0.0206 |
| attn8 | 2.0 / 6.5 / 1.9 / 11.6 | **8.8e-2, 2.8e-2, 9.7e-2**, 1.4e-2 | 0.0832 |
| attn20 | 1.7 / 1.3 / 1.7 / 4.1 | 9.5e-2, **1.5e-1**, 1.2e-1, 4.9e-2 | 0.1158 |

读法：**每层恰好一个（或两个）stream 明显更差，其余低一个量级**，而合成 relRMS 落在
「独立 fp64 参考对 SGLang 的距离」（§12.12：L1 是 0.87 / 0.98 / 0.92 的等边三角形）上。
mHC 的 4 个 stream 是冗余表示，模型只用合成量；单流差异在 `hc_post_combine` 的 4×4 组合里相互抵消
（`attn0_coef` 的组合矩阵第 2 行 `[0.1229, 0.8989, 0, 0]`、第 3 行 `[0, 0.0909, 0.9940, 0]` 这种结构）。
**所以按单 stream 判「谁的错」会得出错误结论** —— 只有合成量有意义。

`A-B = 0.0000` 在 L0/L1 只说明 **SGLang 对自己是确定的**，不说明它更准（§12.12 的等边三角形）。

### 12.25 teacher forcing 仪器有两个 bug（修好后立刻到 30/32 的天花板），并由此定出 L0 的种子机制

**仪器 bug（两个，都在我这边）**

1. `step()` 里写的是 `teach_row(sub, L)`，但 `teach_row` 内部判 `kind == "ffn"` ——
   `"moe"` 永远不等于 `"ffn"`，于是**每个 MoE 都读到 `mpost(L)` 而不是它需要的 `apost(L)`**
   （两者 relRMS 0.35–0.60）。改成传 `kind`。
2. `DSV41_TEACH_KERNEL_X` 的默认是「host 重算 x」，而 `apply_pre` 会覆盖本该由 `hc_post_combine`
   写的 x（已在代码注释里记录过的 50–60% 偏差）。默认改成内核自己的 x（`=1`）。

修好前那三行数字（0.5156 / 0.5596）看起来像"内核有问题"，其实全是仪器的锅：

| 运行 | ovl A | ovl B | 落在 A∪B | mean\|d\| vs A | vs B |
|---|---|---|---|---|---|
| SGLang B（地板） | 30/32 | — | 30/32 | 0.1867 | — |
| 自由链 | 27/32 | 28/32 | 29/32 | 0.3851 | 0.3179 |
| teach host-x（2 个 bug） | 27/32 | 28/32 | 29/32 | 0.5156 | 0.5034 |
| teach kernel-x（1 个 bug） | 26/32 | 27/32 | 28/32 | 0.5596 | 0.6211 |
| **teach kernel-x（已修）** | **28/32** | **29/32** | **30/32** | **0.3377** | 0.3225 |

**修好后的全强制运行直接落在 30/32 的天花板上**（即参考对自己的复现率）。注意它是**用了参考残差**的
（teacher forcing），所以不是"优化"，而是**归因仪器**。

**逐子层自身误差（参考输入下，`teach_intrinsic.py`）**

| | 值 |
|---|---|
| 中位数 | **0.00810** |
| 均值 | 0.00947 |
| 最差 | `moe19` 0.03587 |
| 最差点里参考自身 A-B 散布 | 0.114（同层） |

**40 层里没有任何一个子层的自身误差超过参考自身在该层的散布**（最好的一批甚至小 10 倍以上，例如
`attn4` 0.00629 vs A-B 0.02069）。MoE 普遍比同层 attention 差约 2 倍，符合"MoE 多一层路由"的预期。

**L0 的种子机制（0.00265 的来历，已解到公式级）**

`attn0` 的输入是逐 bit 相同的 embedding，所以它的 0.00265 完全是自己产生的。按内核的系数布局
（`pre = coef[0:4]`、`post = coef[4:8]`、`comb = coef[8:24].reshape(4,4)`）解析出来，mHC 的结构是：

| 输出流 j | 主导项 | post_j |
|---|---|---|
| 0 | 0.870·R₀ | 0.0007 |
| 1 | 0.899·R₁ | 0.0005 |
| 2 | 0.994·R₂ | 0.0000 |
| **3** | 0.905·R₃ **+ 0.136·s** | **0.1362** |

**只有 stream 3 掺入了子层输出 `s`**，这正是"每层只有一个 stream 偏差大"（§12.24）的原因 ——
不是那个 stream 算错了，而是**它是唯一承载子层输出的通道**。

于是可以反解 `s`：`s = (y₃ − Σᵢ comb[i][3]·Rᵢ) / post₃`。结果：

```
s 由参考输出反解 : rms 0.498006
s 由内核输出反解 : rms 0.498088
   relRMS 5.079e-03    corr 0.999987117    median|d| 8.96e-04
```

**5.08e-3 ≈ 0.5 个 bf16 ulp**（该量级 ulp 相对值 3.9e-3）。也就是说 `s` 的 **fp32 累加差 1 ulp，
把约一半元素的 bf16 舍入翻到了邻居格点**。这就是 §12.20 的 `tcgen05` vs `mma.sync` 归约树差异
在 L0 的表现；软件层不可消除。

**`hc_post_combine` 的算术已逐运算审过**（`tasks/hc_post_combine.cuh:275-280`）：

```c
float a = __fmul_rn(comb[0][j], rik(0));
for (i...) a = __fadd_rn(a, __fmul_rn(comb[i][j], rik(i)));   // 串行 i=0..3，分离 mul/add（非 FMA）
ob[k] = __float2bfloat16(__fadd_rn(__fmul_rn(post[j], s[k]), a));   // 唯一一次 bf16 舍入
```

与 `model.py:967` 的 `post*x + sum(comb*residual, dim=2)` **结构一致**（都是分离乘加、最后舍一次）。
用 fp64 从 `(R, s, coef)` 重建内核写的 4 个流，能复现到 bf16 舍入以内（j=2 `1.1e-6`、
j=0/1 `1.6e-3`≈0.4 ulp、j=3 由构造精确）。**组合公式不是误差源。**

**系数的 1-ulp 差异也被排除了（换系数实验）**。`hc_coef` 的 24 个 fp32 系数与 golden 的
`L00_ct_pre/post/comb` 不是逐 bit 相同（最大 52 ulp，但都落在 1e-5 量级的小系数上；主导项
`comb[3][3]=0.905` 差 2 ulp、`comb[2][2]=0.994` 差 1 ulp）。用内核同款 fp32 分离乘加算术做替换：

| 变体 | relRMS vs SGLang | 逐 bit |
|---|---|---|
| 内核系数 + 内核 `s`（实际） | 0.00265 | 17596/20480 |
| **只换系数** | **0.00265**（一位没变） | 17596/20480 |
| **只换 `s`** | **0.00012** | 20406/20480 |

**系数贡献为零，`s` 贡献 100%。** 那 1–52 ulp 全部落在 bf16 粒度以下（最大的也只相当于 1 个 bf16 ulp 的
6e-5），一个元素都翻不动。

**漏斗（最终结论）**

```
fp8 GEMM 的硬件归约树差 1 fp32 ulp (§12.20)
  -> 子层输出 s 的 bf16 舍入翻转约一半元素 (§12.25)
  -> mHC 合成 y 在 L0 差 0.00265
  -> 逐层放大（80 个子层，7 层路由平局翻转）
  -> 末端残差 0.0751，参考自身 A-B 是 0.0792   <- 已在参考的散布带内
```

所以「27 → 30」不是修 bug：**没有任何一层是坏的**，每一层都比参考自身更接近数学真值。剩下的差距是
bf16 舍入边界的随机翻转 + 参考自己的抖动。

### 12.26 天花板实测：把第 39 层换成 fp64 精确实现，logits 误差**一点没变**

§12.25 的 last-layer 观察值得追到底：全强制（每个残差都是参考的）时 logits 仍差 0.3377，而自由链是
0.3851 —— 40 层累积只贡献 0.047。第 39 层 MoE 的 `post` 很大（`[0.84, 0.03, 1.20, 0.65]`，L0 是
`[0.0007, 0.0005, 0.0000, 0.136]`），所以它自己的舍入直接进头部。

于是把第 39 层**整个换掉**：用内核自己的 FFN 输入（`moe39_x.bin`，不是 host 重算的）、同一套路由、
checkpoint 权重，在 **fp64** 里重算 SwiGLU（`moe_expert_ref.expert_raw`，§12.16 已证明它 == checkpoint
数学），再走 `hc_post`（保持 bf16，如 `model.py` 的 `y.type_as(x)`）和头部：

| 第 39 层的算法 | mean \|dlogprob\| vs golden | 说明 |
|---|---|---|
| **我们内核**（控制组） | **0.3460** | 实际链路 |
| fp64 精确 MoE，含两处 `act_quant` | **0.3461** | 数学上精确 |
| fp64 精确 MoE，不含量化 | 0.3464 | 完全不量化 |

**三条完全一样。** 也就是说：**0.346 不是我们的误差**，而是「golden 的 `L39_apost` → 任何精确的第 39 层」
与 golden 自己写出的 logits 之间的距离 —— 即**参考自己在最后一层的舍入噪声**。任何实现到这里都得付这笔钱。

头部配方在同一次运行里被验证过：`head_in = norm(bf16(collapse(mpost40, pre)), gamma=norm.weight)`
能复现内核自己写的头部输入到 `relRMS 1.37e-5 / corr 1.000000`（`pre` 取 `moe39_coef` 前 HC 个 fp32）。

**所以自由链的 0.3851 只比地板 0.346 高 0.039**，而这 0.039 就是 80 个子层的 bf16 舍入在 40 层里累积
出来的量。要把它抹掉等于要求逐 bit 复现（§12.20 已证不可得）。

**top-32 交集同理**：天花板 30/32，自由链 28/32（对 A）、29/32（对 B）、30/32（落在 A∪B 内）。
差的这 2 个就在 §12.24 的并列带里。

**独立旁证**：全 40 层的 MoE 表（`moe_table.py --quant`，y 级、同 x 同路由 vs checkpoint 数学）给出
`relRMS` 均值 0.01530 / 中位数 0.01476 / cos 均值 0.9998771。**其中最准的一层正是 L39**
（0.00326，cos **0.9999948**）—— 直接进头部的那一层我们做得最好，可它的 logits 误差仍是 0.346。
两条独立路径指向同一个结论。

---

## 9. 变更记录

| 文件 | 内容 |
|---|---|
| `demo/deepseek_v4_1_sglang/y_table.py` | **新**：80 个子层逐 kernel 的 `y` 打分（`y = (R' − combᵀR)/post` 四流最小二乘反解）|
| `demo/deepseek_v4_1_sglang/moe_expert_ref.py` | **新**：专家级 MoE 参考（fp4/fp8 分派、块状与逐行两种 scale 布局、SwiGLU 钳位顺序），对 `mpost_x` cos 0.9991–0.9995；`--from-chain` 用同一个 x 对比内核自己的 y（cos 0.9993–0.9996）。**11-27 关键修正**：`act_quant_swiglu_task` 在 fp8 量化前还有一次 `bf16r(silu(g)·u)`，补上后逐专家 relRMS **归零**、40/40 层 cos ≥ 0.9999（§12.16）；`act_quant(kbits=True)` 用内核的 `ceil_log2_bits` 位精确形式 |
| `demo/deepseek_v4_1_sglang/stage_probe.py` | **新**：MoE 分阶段逐字节判定（act_quant / topk_route / w13+SwiGLU+quant），修了自写 e4m3 编码器的进位与符号位两个 bug |
| `demo/deepseek_v4_1_sglang/quant_probe.py` | **新**：两级 act_quant 的净影响（3.28% → 1.95%）|
| `demo/deepseek_v4_1_sglang/variants.py` | **新**：MoE 实现变体扫描（`quantize(w·h)` / g,u 是否 bf16 / 钳位顺序 / w2 输出是否 bf16 / h 用 fp8·bf16·fp32）|
| `demo/deepseek_v4_1_sglang/moe_table.py` | **新**：40 层 MoE 保真度表（relRMS / cos / 是否 ≥0.9999），`--quant --kbits` 为 like-for-like 口径 |
| `demo/deepseek_v4_1_sglang/bitexact.py` | **新**：运算级逐 bit 判定 —— 用内核 `pairs` 的 fp32 路由权重，逐 (expert, rank) 核对 `routed_rows` 与 `sd_part`；结论 42/48 格 100% 逐 bit、全部 max\|d\| 7.45e-09、shared cos 1.0000000000（§12.17）|
| `demo/deepseek_v4_1_sglang/attn_nodes.py` | **新**：attention 逐节点判定 —— 直接读内核写的 buffer（`x_q`/`qr_q`/`oa_q` 三个量化节点 + `wq_a`/`wq_b`/`wkv`/`wo_a`/`wo_b` 五个 GEMM），解 `zq_bytes` 的 scale 块布局；结论 3 个逐字节相同、5 个 cos 恰好 1.0、worst 7.1e-7（8 rank），§12.18 |
| `demo/deepseek_v4_1_sglang/attn_core.py` | **新**：按官方 `kernel.py sparse_attn_kernel` 复现两段 64-key online softmax（per-block max / `O·sc + bf16(P)·K`），证明 `o` 的 0.0085 不是分块结构造成的（1.00x），§12.18 |
| `demo/deepseek_v4_1_sglang/tilelang_check.py` | **新**：**跑官方 tilelang `sparse_attn`**，喂 megakernel 自己 buffer 里的 q/K，判定 `o_q` —— 32768/32768 逐 bit 相同（8 rank，两种 rope 都对），§12.19 |
| `demo/deepseek_v4_1_sglang/tilelang_gemm.py` | **新**：**跑官方 `fp8_gemm_kernel`**（block 32、e8m0、fp32 累加），对四个 fp8 GEMM 节点判定：fp32 ~1 ulp，**接口值 bf16 88063/88064**；并量化两边距精确值都 ≤0.2 ulp（megakernel 四个里三个更近），§12.19 |
| `demo/deepseek_v4_1_sglang/gemm_reduction.py` | **新**：把 fp8 GEMM 的 fp32 差异拆成「32 项点积形态 × 外层累加结构」12 种模型。判定：参考 = 32-dot 精确 + 逐块 fp32 顺序累加（**1253/1280 可复现**）；megakernel 用任何 IEEE 模型都复现不了（最高 683/1280，即正确舍入值本身）→ 差异在 `tcgen05 block_scale` 数据通路内部。§12.20 |
| `demo/deepseek_v4_1_sglang/teacher_table.py` | **新**：teacher-forced 逐子层判据表（`{sub}_teach_out.f32` vs 金标，附参考自身的 6-vs-7 地板）|
| `demo/deepseek_v4_1_sglang/full_decode.py` | `DSV41_TEACH_KERNEL_X=1`：teacher 模式不覆盖 FFN 的 `x`（让内核自己从金标残差算）；附实测说明为何 MoE 的 teacher 判据不可用 |
| `demo/deepseek_v4_1_sglang/FINDINGS_correctness.md` | 本文 |
| `include/mirage/static_megakernel/core.cuh` | `STATIC_TP_BARRIER`：内核内跨 rank 起点栅栏（原本是死代码，现由 build 定义）；`DSV41_RESET_DEBUG` 的 reset 记账 |
| `include/mirage/static_megakernel/host.cuh` | `@prime`：重新武装 by_reader 缓冲区与 stamps；`DSV41_RESET_DEBUG` 的 reset 列表 dump |
| `include/mirage/static_megakernel/runtime.cuh` | `DSV41_MBAR_DEBUG`：有界等待并报告未见到的 mbarrier |
| `include/mirage/static_megakernel/tasks/{gemm_fp8_tile,sparse_attn}.cuh` | `DSV41_SA_QPOLL_DEBUG`：有界等待并打印轮询到的字节 |
| `demo/deepseek_v4_1_sglang/dsv41_static_layer.py` | `tp > 1` 时默认加 `-DSTATIC_TP_BARRIER`（`DSV41_STATIC_TPB=0` 关闭）；`DSV41_STATIC_SERVING=0` 走 host reset 路径 |
| `demo/deepseek_v4_1_sglang/dsv41_static_hook.py` | `TPBAR` + `_tpb`；banner 打印 `TPBAR=` / `TPB=`；两者皆 0 且纯静态链时打 WARNING |
| `demo/deepseek_v4_1_sglang/full_decode.py` | 单进程 40 层链；mHC `S0 = embed.repeat(HC)`；logits 取 rank 0 全量；**11-27：x 接线（只有 L0 用 `X`）+ Engram 注入只写 row 0 + `DSV41_EG_ZERO` 诊断**；**11-27 收尾：`hc_pre()`/`apply_pre()` 取代 `sublayer_x()` —— pre_mix 从生产 kernel 的 `hc_coef` 读，L0 attention 用 identity（§12）** |
| `include/mirage/static_megakernel/tasks/sparse_attn.cuh` | **11-27：`BF16O`（params[7]）—— o 直接写 bf16，跳过 act_quant**（§10） |
| `python/mirage/mpk/static_megakernel.py` | **11-27：`sparse_attn_layer(output_bf16=)`，`act_quant_layer` 接受 `gemm_tile` 的 slots**（§10） |
| `demo/deepseek_v4_1_csa2/{demo,weights}.py` | **11-27：`DSV41_WO_A_BF16`（默认 1）—— wo_a 反量化成 bf16 + `gemm_tile_layer`**（§10） |
| `demo/deepseek_v4_1_sglang/dump_hook.py` / `dump_prefill.py` / `dump_boot/` | 从 TP worker 内部 dump prefill 的 KV pool / 寻址表 / 每层残差 |
| `demo/deepseek_v4_1_sglang/attn_np.py` | **新增**：单个 swa attention 子层的参考重算（权重+池+RoPE+sink softmax+o·wo_a·wo_b），`--gemm` 用链自己的 o 走一遍 GEMM（§12.8） |
| `demo/deepseek_v4_1_sglang/align3.py` | **新增**：链 vs 两次独立 SGLang dump vs SGLang 对自己 的三方对齐表（§12.3） |
| `demo/deepseek_v4_1_sglang/attn64.py` | **新增**：单个 swa attention 子层的 **fp64** 参考（精确反量化 + 池里的 K + sink softmax + bf16 `o` 直进 bf16 `wo_a` + wo_b），正向比 `R'`；用于判定 0.2–0.9% 归谁（§12.9） |
| `demo/deepseek_v4_1_sglang/moe_audit.py` | **新增**：全 40 层路由翻转审计 —— 两边各按 `Block.forward` 重建 FFN 输入，只算 gate，报告 top-6 集合是否改变与第 6/7 名相对间隔（§12.10）；`--save-x` 写出的就是 `moe_ref.py --x` 要的口径 |

| `demo/deepseek_v4_1_sglang/moe_ref.py` | 加 `--save`（把 routed+shared 写出来）。**11-27 修两个真 bug**：`--x` 是 1 维导致 `[0]` 取标量再广播（`reshape(1,H)`）；权重漏了 top-k 归一化（`ModelArgs.norm_topk_prob` 默认 `True`）。修后量级对上，但 expert 级 cos 仍只有 0.23，别用绝对数字（§12.11） |
| `demo/deepseek_v4_1_sglang/l2_check.py`、`golden_all.py`、`golden_set.py`、`golden_forced.py`、`compare_golden.py`、`e2e_decode.py`、`bisect_repeat.sh` | 逐 sublayer 对照、golden 生成与比对、噪声底 |
| `demo/deepseek_v4_1_sglang/teach_intrinsic.py` | **新增**：teacher forcing 下的逐子层**自身**误差表（每层读参考残差，误差不再混入上游漂移），并列参考自己的 A-B 散布作标尺（§12.25）。**注意 `moe{L}` 的对照物是 `L{L+1}_mpost`**（`align3.py:57`），不是 `L{L}_mpost` |
| `demo/deepseek_v4_1_sglang/last_layer_ceiling.py` | **新增**：天花板实测 —— 用内核自己的 FFN 输入 + 同一路由，在 fp64 里重算第 39 层（`quant` 开/关两档），接头部比 logprobs。三条（内核 / fp64+quant / fp64）都是 0.346，证明该误差是参考自己最后一层的舍入噪声，不是本实现的（§12.26） |
| `demo/deepseek_v4_1_sglang/full_decode.py` | **11-27 加 `DSV41_TEACH_FROM=L`**（第 L 层起才强制参考残差，其余自由跑 —— 归因用；40 = 自由链，0 = 全强制）；**修 teach 的 bug**：`teach_row(sub, L)` 传的是 `"moe"` 而函数内判 `"ffn"`，导致每个 MoE 都读 `mpost` 而非 `apost`（relRMS 0.35–0.60）；`DSV41_TEACH_KERNEL_X` 默认改为 1（host 重算 x 会覆盖 `hc_post_combine` 写的 x）。修后 teach 全强制 = 30/32（天花板）、mean\|d\| 0.3851 → 0.3377（§12.25） |
| `.gitignore` | `demo/deepseek_v4_1_sglang/1/`（`MIRAGE_STATIC_BUILD_CACHE` 是目录，被误写成 `1` 的产物） |

### 12.27 权重保真：内核持有的张量是否就是 checkpoint 自己的字节（fp8 仍 fp8、bf16 仍 bf16、fp32 仍 fp32）

**缘起**：`dump_hook.py:85/99` 把残差写成 fp32 —— `np.array(rec["layers"][k].numpy(), dtype=np.float32).tofile(...f32)`。
所以所有 `.f32` dump、以及任何“看起来全是 fp32”的观察，**只说明激活被存成了 fp32，与权重的精度无关**。权重必须在自己
的 dtype 上比字节；跨 dtype 比（bf16 抬成 fp32、fp8 先反量化）正好抹掉要查的那个误差。

**两个脚本，两种口径**（都严格在各自 dtype 上比，绝不做隐式转换）：

| 脚本 | 比什么 |
| --- | --- |
| `weights_check.py` | 加载器（`csa2/weights.layer_tensors`）的**输出**：fp8 用 uint8 字节；e8m0 scale 用 `w8_scale_chunks` 的逆置换；bf16 用 bf16 位型；fp32 用 fp32 位型；wo_a 的 fp8→bf16 反量化要求**零舍入**（e4m3 × 2 的幂最多 4 位有效数字，落 bf16 必须精确） |
| `held_weights_check.py` | **内核真正持有的张量**：按 `full_decode.py` 的口径搭出每 rank 的权重字典，逐个对回 safetensors。fp4 MoE 是重排 + pad 的布局，用 `unpack_pieces` 把两侧**独立解成 fp64 再逐值比**（解码是字节的无损函数，故等价于比字节）；pad 行必须全 0、pad scale 必须 127 |

**结果**：`weights_check.py` 7 层 **119/119**；`held_weights_check.py` 7 层 **150/150**（含 indexer / compressor / Engram），
0 个 MISMATCH、0 个 DTYPE、0 个 INEXACT。覆盖 swa / full_r2 / full_r1 / reuse / reindex 全部变体，TP8 rank0。

**逐类结论**：

- **fp8 e4m3**（`wq_a`、`wkv`、`wq_b`、`wo_b`、`indexer.wq_b`、`shared_experts.*`、Engram `wkv`）：内核里就是 uint8 的
  e4m3 原始字节，**逐字节相同**（最大一例 `L0.engram.wkv` 19 660 800/19 660 800 bytes）。
- **fp4 e2m1**（MoE routed `w1/w2/w3`）：checkpoint 存成 `int8` 打包对；内核的 `w13`/`w2` 是 64 行 item + `pad_rows` +
  `piece_chunks` 的重排。独立解码后逐值相等（`w13` 6 553 600/6 553 600、`w2` 3 932 160/3 932 160，每层 3 个专家含
  e0/e1/e192）。
- **e8m0 block scales**：`w8_scale_chunks` 是**纯置换 + 32 行复制**，逆置换后字节完全一致 —— 这个变换无损。
- **bf16**（`q_norm`、`kv_norm`、`compressor.*`、`indexer.wk/k_norm`、`indexer.weights_proj` → pad 到 128 行为 **0**、
  `gate.weight`、`head.weight`）：bf16 位型相同；`wproj` 的 491 520 个 pad 值全为 0。
- **fp32**（`attn_sink` 8/8、`gate.bias` 384/384）：本来在 checkpoint 就是 fp32，位型相同。
- **唯一的“fp32 派生量”**：Engram 的 `eg_qk = q_weight.float() * k_weight.float()`。这**不是**上转：官方
  `inference/model.py:356` 就是 `weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product`，
  且两个 bf16 相乘在 fp32 里是**精确**的（8+8 ≤ 24 位尾数）。我们与参考逐位相同（20 480/20 480）。
- **非权重的 fp32 buffer**（不要误判为权重被抬精度）：`logits`（输出，`[T, 129280]` fp32）、`cmp_state`（compressor 的
  per-token ring `kv | score`，运行时状态）、各 dump 的 `.f32` 残差。

**结论**：内核的权重集**就是 checkpoint 自己的字节**，没有任何张量被上转成 fp32 再使用。激活里出现的 fp32 只来自
dump 格式与上述三个 buffer。因此前面 §12.19–12.26 的所有数值分歧都不可能由“权重走了 fp32”解释。

| `demo/deepseek_v4_1_sglang/weights_check.py` | **新增**：加载器输出的权重保真核对（fp8 bytes / e8m0 逆置换 / bf16 位型 / fp32 位型 / wo_a 零舍入反量化），7 层 119/119（§12.27） |
| `demo/deepseek_v4_1_sglang/held_weights_check.py` | **新增**：**内核实际持有**的权重核对（含 fp4 MoE 重排的独立 fp64 解码、indexer/compressor/Engram、pad 必须全 0），7 层 150/150（§12.27） |

### 12.28 参考自身的逐层不确定性：L0/L1 逐位确定，L2 起 s 就差 1.34%

**方法（这是关键，避免再从残差结构"反推"）**：`dump_hook.py` 记录的 `L{l}_ct_resid_fp` 是**系数所依据的那个残差的指纹**
（4 个 stream 的范数 + 前 8 值，12 个数）。用它把每个 `_ct_*` 精确对回某个残差 dump，就确定了 mHC 方程的入口 `R_in`，
不用猜。再解
$$y_j = post_j\, s + \sum_i comb[i][j] R_i \;\Rightarrow\; s = \frac{y_j - \sum_i comb[i][j] R_i}{post_j}$$
取 $j=\arg\max|post|$（信噪比最好的那个 stream）。**mix 模型在全部层闭合到 1.1e-3 relRMS**（`s` 存成 bf16 的重建地板），
所以解出的 `s` 可信，而不是拟合出来的。

指纹给出的入口（注意 `_mpost` 按**它喂给的那一层**命名）：

| 子层 | `R_in` | 输出 |
| --- | --- | --- |
| attn0 | `L00_in`（embedding） | `L00_apost` |
| attn1 | `L01_in` ≡ `L01_eg_y`（Engram 替换后） | `L01_apost` |
| attn L≥2 | `L{l}_mpost`（即 moe L-1 的输出） | `L{l}_apost` |

**`post` 逐层轮转**（先前"误差集中在 stream 3"是 L0 的特性，不是通则）：

| 层 | argmax stream | max\|post\| |
| --- | --- | --- |
| L0 | 3 | 0.1362 |
| L1, L2, L7, L20, L39 | 0 | 0.19 / 0.22 / 0.36 / 0.55 / 1.67 |
| L3 | 1 | 0.2021 |
| L14 | 2 | 0.4987 |

**SGLang 四次独立启动之间，同一个 `s` 的散布**（dec_coef6/7/8，均有 `ct_resid_fp`）：

| 层 | rms\|s\| | s 的 relRMS 散布 |
| --- | --- | --- |
| **L0** | 0.498 | **0（逐位相同）** |
| **L1** | 0.351 | **0（逐位相同）** |
| L2 | 0.360 | **1.34e-2** |
| L3 | 0.312 | 4.27e-2 |
| L5 | 0.361 | 3.67e-2 |
| L10 | 0.416 | 8.00e-2 |
| L15 | 0.701 | 1.06e-1 |

自洽性：L2 的 `post_0=0.2194`、`Δs rms = 1.34%×0.36 = 4.8e-3` → 对 stream 0 贡献 `1.05e-3`，除以 `rms|Y|=0.126`
得 `8.3e-3`，与实测 stream 0 散布 `9.4e-3` 一致。L2 的 `ct_pre/post/comb` 在四次启动中**逐位相同**，`L02_in`、
`L02_mpost` 也逐位相同 —— 所以差异**完全来自 attention 子层输出 `s` 本身**。

**这不是"我们的 bug"，是 `full_r2` 那一层（compressor + indexer）在 SGLang 内部就不确定**：L0/L1 是 `swa`，逐位确定；
从第一个 `full_r2` 层 L2 开始才有散布。可能来源是 indexer 的 top-k 选择、compressor 的分块归约顺序、或 KV 池里
chunked prefill 的排列 —— 但它们都发生在 SGLang 自己的 attention 内部，dump 层面观察不到。

**对目标的直接后果**：任何实现都无法与"这一个 SGLang 结果"在 L2 之后做到 bit-exact —— 没有唯一的 SGLang 答案。
可达的上限是：**L0/L1 逐位相同**，L2 之后落进上表的散布带。我们的逐子层固有误差中位数是 8.1e-3、最坏 3.59e-2
（§12.25），即**在 L2/L3/L5/L10 我们自己的误差比参考自身的散布还小**；从 L15 起参考的散布（1.06e-1）超过我们，
对比值由参考主导。因此"40 层全部 relRMS < 1%"在参考侧就不可能达成。

### 12.29 输入目录喂错：`prefill_row0` 的窗口少了最后一格 KV（"L0 attention 差 21.6%"的假象根源）

**症状**：用 `/tmp/bit_L0` 的 dump 判 `attn0_r0_o_q`，任何 fp64 重放都对不上（0.19~0.30），曾经据此怀疑
circular window 的 `window_loc` 偏移、RoPE position、softmax 归约顺序 —— 全部排除（`window_shift.py` 扫 k∈[-4,4]
无锐利最小值；RoPE position 扫 ±2 亦无）。

**根因**：`/tmp/bit_L0` 那次跑用的 `--inputs-dir /tmp/prefill_eg`，而 `prefill_eg` 软链到 `prefill_row0` —— 那是
**prefill 阶段**的 dump，`seq_lens=31905`。decode 到 `pos=31906` 需要 KV 历史覆盖 position 31905，而那一格在
prefill dump 里**还没写**（全 0）。两个池的 128 行窗口里**只有这一行不同**（其余 127 行逐字节相同），但全 0 的 K
让 score=0，在所有真实 score 为负时 `exp(0-m)` 在分母里塞进一大项，同时少掉一个真实 key，输出被整体拉小。

叠加的第二个错位：`attn64.py` 的 `PRE` 写死 `/tmp/dec_coef8`，于是"拿 A 的池验 B 的输出"。

**判定证据**（用 `y_gold = (apost - comb·in)/post` 反解 SGLang 自己的 L0 attention 输出）：

| 重放用的池 | vs `y_gold` |
|---|---|
| `/tmp/dec_coef8`（decode dump, seq_len 31906） | **1.83%**，cos 0.99983 |
| `/tmp/prefill_row0`（prefill dump, seq_len 31905） | **49.4%** |

换成正确输入（`--inputs-dir /tmp/dec_coef8`）后：`attn0_r0_o_q` 对重放 **4.47%**（原 19.4%），
L0 端到端 `y` 对 `y_gold` **0.454%**，cos 0.9999897，幅值 0.49806 vs 0.49801（反解自身的 bf16 噪声地板是 0.23%）。

**结论：模型结构与 kernel 的计算都是对的；之前"L0 attention core 差 21.6%"是工具链 artifact。**

### 12.30 逐层比对的正确配对：`_mpost` 的编号是"在哪层跑的"，且参考里没有 MoE 的系数

两个 mHC 事实，都能从源码直接读出，不必从 dump 反推：

1. `deepseek_v4_mhc.py:886` 把**上一层 post 与下一层 norm 融进同一个 kernel**
   （`mhc_mega_boundary(y, residual, pre, post_mix, comb, next_hc.fn, next_hc.norm.weight, ...)`）。
   `dump_hook.py` 的 `_apost/_mpost` 都记在 `_gate["lid"] = self.layer_id`（**当时正在执行的层**），于是
   layer L 的 MoE post 落成 `L{L+1}_mpost`。layer 0 更特殊：其 MoE 输出是 layer 1 的 engram 输入 → `L01_eg_x`。
2. `L{L}_ct_pre/post/comb` 是**attention** 的 triplet（`attn0_coef` 对 `L00_ct_*` 到 **1.8e-7**）。
   MoE 的 triplet 不经过被 hook 的 `_compute_triplet`，**参考 dump 里没有**；拿 `moe{L}_coef` 去比 `L{L+1}_ct_*`
   会得到 0.24~1.67 的假差异。

用 `hc_{\{attn,ffn\}}_fn/scale/base` 按官方公式（`pre=sigmoid`、`post=2·sigmoid`、`comb` 走 `hc_sinkhorn_iters`
次 Sinkhorn）自算 triplet：对 `L00_ct_*` 复现到 **1e-8**；对链的 `moe{L}_coef` 是 L0 **1.9e-4** → L2 2.1e-3 →
L20 1.6e-2，**随残差误差单调退化，没有系统性偏差**。

### 12.31 40 层端到端（正确输入）与 logits

正确输入的 40 层跑（`/tmp/chain_ok`，`--dump-all attn --rms`）用 `layerwise.py`：

* attn 侧：L0 **2.65e-3** → L7 3.67e-2 → L39 8.84e-2；moe 侧：L0 6.88e-3 → L39 无参考（`L40_mpost` 不存在）。
* L8..L39 共 62 个点里只有 L18 一个超出 SGLang 自身散布带；L39 我们 8.84e-2 **小于**参考自身 4 次运行的
  1.09e-1 散布。误差在 L8 之后**饱和**（不是持续放大），这是"数值不同但数学等价"的签名。
* L0 的 ulp 分布：86.1% 逐位相同，中位数 **1 ulp**，p99 72 ulp —— bf16 存储 + 不同 fp32 累加顺序的地板。
* logits：argmax **343 与 golden 一致**；top-32 overlap 27/32，mean|dlogp| **0.342**。128/129280 全在
  bf16 网格上（`== 自身 bf16 舍入` 129280/129280），两侧都是精确 bf16，不存在量化环节。
* 最终 `x` 对参考路径 relRMS **5.95%**，cos 0.99823。用 iid 扰动标定灵敏度：eps 0.09 → 0.17、eps 0.12 → 0.24；
  真实误差是有结构的（集中在权重敏感的通道上），所以 0.342 同量级合理。

**待办**：multi-step（压缩 cache 推进）、engram gate、L2 indexer score 尚未验证。

### 12.32 多步 decode：10 个 greedy 步**逐个 token 命中一个 SGLang 运行**，而 SGLang 自己几次运行轨迹不同

`/tmp/dec_coef8/layers/step1..step10` 是同一条 row 连续 10 个 decode 步的 dump。每步的输入 token 不必猜：
该步的 `L00_in[0]` 就是 `embed.weight[token]`（max|d| 实测 **0.0e+00**）。于是参考侧的轨迹可以直接读出来。

链用 `--steps 9 --greedy` 跑出 step1..step10（`/tmp/chain_m10`），`multistep.py` 逐层比：

| step | 输入 token | L0 attn | L2 attn | L20 attn | L39 attn | argmax vs 参考 |
|---|---|---|---|---|---|---|
| 1 | 4588 | 2.65e-03 | 2.06e-02 | 1.16e-01 | 8.84e-02 | **MATCH** |
| 2 | 343 | 3.19e-03 | 2.61e-02 | 7.97e-02 | 7.64e-02 | **MATCH** |
| 3 | 6333 | 2.99e-03 | 2.31e-02 | 1.39e-01 | 1.19e-01 | **MATCH** |
| 4 | 294 | 2.42e-03 | 2.90e-02 | 1.26e-01 | 9.99e-02 | **MATCH** |
| 5 | 270 | 2.62e-03 | 3.99e-02 | 1.41e-01 | 1.16e-01 | **MATCH** |
| 6 | 16062 | 2.78e-03 | 2.55e-02 | 1.78e-01 | 1.12e-01 | **MATCH** |
| 7 | 21418 | 2.85e-03 | 2.15e-02 | 1.50e-01 | 1.15e-01 | **MATCH** |
| 8 | 682 | 3.08e-03 | 2.74e-02 | 1.33e-01 | 1.05e-01 | **MATCH** |
| 9 | 63356 | 2.73e-03 | 3.05e-02 | 1.25e-01 | 1.14e-01 | **MATCH** |
| 10 | 51773 | 3.05e-03 | 2.68e-02 | 9.15e-02 | 1.17e-01 | (参考无 step11) |

**误差在 10 步里完全持平，没有任何累计增长**（L0 一直在 2.4e-3~3.2e-3，L39 一直在 8.8e-2~1.2e-1）。
这直接证明三种缓存推进都在 kernel 内部做对了：

* `window_cache` 的 128 槽环形写 —— L0 是纯 `swa`，写槽错位会立刻 O(1)，而它 10 步都在 3e-3；
* `compress_cache` + `cmp_state`（ratio 2 的压缩 KV 与半块状态）—— L2 是 `full_r2`，10 步都在 2.1e-2~4.0e-2；
* `compress_idx` 每步重做 indexer top-k —— 同上。

`advance()` 只搬 `positions` 与页表，数据全由 `kv_post` / `ckv_post` / `compressor` / `indexer_topk` 写。

**由此推翻待办里的两条旧判断**：（a）"`--greedy` 只移动窗口表，压缩 cache 不推进"—— 不成立，缓存确实推进；
（b）"token 对不上 golden 是链的错"—— 不成立，见下。

#### 谁在偏离：SGLang 自己

同一 row、同一参数下 4 个 SGLang 运行的 token 轨迹（每步输入 token，从各自 dump 的 `L00_in` 读出）：

```
golden     : [4588, 343, 6333, 294, 270, 4245,  344,  12070, 682,   1]
dec_coef8  : [4588, 343, 6333, 294, 270, 16062, 21418, 682,  63356, 51773]
dec_coef6  : [4588, 343, 6333, 294, 270, 16062, 21418, 682,  63356, 51773]
dec_coef7  : [4588, 343, 3050, 7082, 638, 868,   795,   2411, 70943, 17]
```

* `dec_coef6` 与 `dec_coef8` **10 步完全一致**，但在第 5 步就离开了 golden（16062 vs 4245）；
* `dec_coef7` 在第 2 步就分叉（3050 vs 6333）；
* golden 与 `dec_coef7` 只重合 2/10，与 `dec_coef6/8` 重合 5/10。

**我们的链 10 步全部命中 `dec_coef6`/`dec_coef8` 的轨迹** —— 与 §12.28「L2 起参考自身不确定」完全吻合：
不确定性不只是数值噪声，它会改变 greedy 的 token 选择。所以"与 golden 逐 token 相同"不是可达目标（golden 只是
其中一个 SGLang 实例的一次结果），正确标准是**命中某一个 SGLang 运行**，而我们做到了 10/10。

第 5 步那个分叉本身是一个接近平局：我们的 logits 里 `16062` 是 -0.7635、golden 的 `4245` 是 -1.1385，
只差 **0.375 nats**。这是 SGLang 内部 ~1e-2 级别的数值分歧足以翻转的量级。

### 12.33 Engram gate 正确：`R''−R'` 在每一流的**绝对**误差都是 5–7e-4

Engram 位于 MoE 与下一层 attention 之间，它的 n-gram 嵌入行是**输入**（表查找由 SGLang 做），所以这是唯一两侧
读同样字节的子层。`engram_check.py` 比 `moe{0,13}_streams_eg` 与 `L{01,14}_eg_y`。

SGLang 的公式（`layers/engram.py:1698`）：

```
out = h + gate_i * value          gate_i = sigmoid(copysign(sqrt(|dot_i|), dot_i))
dot  = (h * (q_weight*k_weight) * key_i).sum(-1) * rstd * dim**-0.5
```

`value` 对所有 hc 流相同，所以 `delta_i = gate_i * value` 应当**逐流成比例**。实测参考侧确实如此：
`cos(delta_i, delta_3)` = 0.988/0.997/0.973，与 gate 比值 0.043/0.065/0.045 自洽。

把继承的残差误差（`moe{host}_streams_out − L{L}_eg_x`）扣掉、只看 gate 自己的贡献：

| engram | 逐流 gate relRMS | 逐流 cos |
|---|---|---|
| L01 | 0.0040 / 0.0030 / 0.0835 / 0.0870 | 0.999992 / 0.999997 / 0.9965 / 0.9962 |
| L14 | 0.1962 / 0.0607 / 0.1444 / 0.0104 | 0.9878 / 0.9982 / 0.9896 / 0.99999 |

**关键在绝对量**：L14 的 stream 3 信号 6.5e-2、误差 1.04% → 绝对 6.8e-4；stream 0 信号 2.8e-3、误差 19.6%
→ 绝对 5.5e-4。**每一流的绝对误差都是 5–7e-4**，即 gate 本身正确，只是信号小的流相对误差被放大。

**一个必须避免的误读**：L14 的 delta 看似"摊到所有流"，其实是继承误差 —— `moe13` 的 R' 逐流误差是
1.79e-02 / 1.21e-02 / 2.69e-02 / **1.47e-03**（stream 0/1/2 差 6.5%，stream 3 只差 0.15%），把它加到 delta 上
就伪装成"gate 在所有流都写了"。`engram_check.py` 现在把这个分离直接打印出来。

### 12.34 Indexer 审计：L2 的 top-512 在截断处是**精确并列**，18/512 的取舍由实现定义

L2 是唯一"超出参考噪声带"的早期子层（attn 2.06e-02 vs band 6.35e-03），而 `ref_spread.py` 早已记录：SGLang 自己在
L0/L1 **逐位可重复**，L2 起不可重复，且差异落在 **L2 的 attention 内部** —— `L02_st`/`L02_apost`/`L02_apost_x` 变，
而 `L02_in`/`L02_mpost`/`L02_ct_*` **完全不变**。输入没变、变的是 attn 里的东西。

`--dump-all attn2` 拿到 `attn2_r*_scores`（`[T, CAP]` bf16，CAP = 16128）后，用我们的 score 向量检验：

```
  排序后:  506 个严格大于 cut，24 个恰好等于 cut = -1.3828125
  top-512 只能取这 24 个中的 6 个  ->  18 个槽（3.5% of 512）的取舍完全任意
  前 520 个 score 只有 106 个不同的 bf16 值
  并列的槽: 2878 4566 4638 4766 4769 5186 5280 6352 ... 15940
```

score 存的是 bf16（`indexer_score_layer`: `out[...] = __float2bfloat16_rn(sum)`），而 score 的量级 ~1.9，bf16 的 ulp
~1.2e-2，所以并列极密。**top-k 在硬并列上是实现定义的**：任何 fp 顺序差异、甚至 SGLang 自己那个本来就不逐位可重复的
L2 score kernel，都会从这 24 个里挑不同的 6 个。

于是三条现象一次解释清楚：

1. SGLang 在 L0/L1 逐位可重复 —— L0/L1 的 `compress_ratio = 0`，**没有 indexer**；
2. L2 起不可重复，且差异只在 attention 的输出侧 —— 变的正是 KV 选择；
3. 我们的 L2 略超带 —— 我们只是那 18 个模糊槽的**另一个同样合法的抽样**。

结论：**indexer 没有 bug**。它复现了 SGLang 的 bf16 score 网格（我们的 score 落在同一批 bf16 值上），歧义来自截断处的
并列，而不是我们的计分或 top-k 逻辑。想再进一步逐槽比对需要 `page_table`（`indexer_score_layer` 把它当图输入，
SGLang 的 `dump_hook` 不记录它），价值不高。

**证据链小结**：L0 attention 端到端 0.454%（§12.29）→ engram 每流绝对误差 5–7e-4（§12.33）→ 10 步 greedy 零增长
（§12.32）→ L2 的歧义是并列而非计分（本节）。**没有发现任何数学错误；残差是"fp32 规约顺序"这颗种子经 40 层放大的
结果** —— 我们的 L0 attention 是 2.65e-3（SGLang 自己是 0，逐位），这颗种子在 L2 被 top-k 并列放大成一次重采样，
到 L8 就落进 SGLang 自身的噪声带（8e-2）里、之后再无区别。

### 12.35 L0 的种子定位到「SGLang 的 decode 与 checkpoint 自带参考实现差 1 个 bf16 ulp」

目标要求的口径是 L0 attention 的良态度量（stream 3，post = 0.1362）。这一节把它追到底。

#### 先把度量做对：`R' = bf16(post[i]·y + Σ_j comb[j][i]·R[j])` 可以**逐 bit 复现**

用参考自己的 `L00_ct_pre/post/comb`、`L00_apost_x`（y）、`L00_in`（R）在 fp64 里算上式再舍到 bf16：

```
  reconstruct R' from ref coef + ref y : relRMS vs L00_apost 0.000e+00     <- 逐 bit 相同
  reconstruct R' from OUR coef + ref y : relRMS vs L00_apost 0.000e+00
  our coef vs ref coef: max|d post| 2.98e-08   max|d comb| 1.19e-07
```

所以 (a) mHC 的公式与 `comb` 的取向（列形式）是对的，(b) 我们的系数是 fp32 精度而非舍入值，
(c) 之前看到的 `0.1362` 是**我自己 `np.round(...,5)` 打印的假象**，(d) `L00_apost` 与 `streams_out`
之间 2.647e-3 的差**完全来自 y**，没有别的环节。

用上式反解出我们的 y（100% 元素在 bf16 网格上精确解出，即每个元素都能原样重建我们的 `streams_out`）：

```
  our y vs ref y: relRMS 4.984e-03  cos 0.999987603   bit-exact 0/5120
  norms 0.498088 vs 0.497986     |d| 的 bf16 ulp 分布: <0.5ulp 1685, 1ulp 1834, 2ulp 615, 3ulp 272
```

#### 我们这一侧是**参考自身的算术**

| 检查 | 结果 |
|---|---|
| 输入 x：`attn0_x` vs RMSNorm(Σ pre_c·L00_in_c)·γ（checkpoint 权重） | **5120/5120 逐 bit** |
| `x_q` / `qr_q` / `oa_q`（e4m3 buffer） | 40960/40960、10240/10240、**8192/8192 逐字节** |
| `o_q` vs 官方 tilelang `sparse_attn`（同样 q、K、sink） | **32768/32768 逐 bit** |
| `wo_b` 的 bf16 接口值 vs 官方 `fp8_gemm_kernel`（同样 `oa_q`） | **8 rank 全 5120/5120 逐 bit** |

即：给定同样的输入，我们的 y **就是** checkpoint 自带 `inference/kernel.py` 给出的那个 bf16 值。

#### 那么差异在 "参考 vs SGLang decode"

用链自己的 `o_q`（已证逐 bit 等于官方）喂不同的算术，看谁能重现 `L00_apost_x`：

| 算术 | relRMS | cos |
|---|---|---|
| fp8 精确反量化 + fp32 累加（`wo_a` 输出不取 bf16） | 9.56e-3 | 0.99995436 |
| **同上 + `wo_a` 输出先取 bf16（`bf16r(x)`，源码里写死）** | **4.10e-3** | 0.99999160 |
| 权重先取 bf16 | 9.56e-3（**与第一种完全相同**：e4m3 只有 4 位有效位，bf16 无损，所以这不是一个可区分的自由度） |
| 组间 bf16 累加 | 1.03e-2 | 0.99994730 |

结论：补上 `wo_a` 输出那一次 bf16 舍入后，参考算术到 SGLang 的 y 是 **4.10e-3**，而我们内核是
**4.98e-3** —— 两者都在 **1 个 bf16 ulp**（相对 3.9e-3）的量级。**没有任何算术变体能到 0**，
包括 checkpoint 自己的参考实现。

所以 L0 的种子**不在我们的流水线里**：它是 SGLang decode 路径的 fp32 归约顺序与参考实现不同的结果，
被 y 边界那一次 bf16 舍入忠实放大成 ~5e-3。这与 §12.19 的结论一致（fp32 那一层本来就不是逐 bit，
只有 bf16 接口值才是），只是这里把它推到了 L0 的端到端。

#### 同时发现：§12.23 的「并列恰好够用」是错的

`indexer_score` 的 `scores` 是 `[T, 16128]`，但 token 只有 `n = (p+1)/R = 15953` 个压缩位置，
**索引 15953..16127 的 175 个槽是填充**，值是 `0.0`（`srt[1:176]` 全是 0 —— 0 比所有真实分数都大）。

- 内核是对的：`indexer_topk.cuh:201` 有 `n = min((p+1)/R, CAP)`，只在前 n 个里选；
- 但 §12.23 的统计把 175 个填充槽算进去了 → 得到 "495 严格更高 + 17 并列 + 恰好 17 个空位"；
- **排除填充后**：`506 严格更高 + 24 个精确并列于 -1.3828125，而 top-512 只剩 6 个空位`
  → 18 个槽的取舍由实现定义。

§12.23 的结论"L2 的 top-k 不是歧义源"因此不成立；正确说法是：**L2 的 top-512 确实有 3.5% 的
实现定义歧义**（16 个不同 bf16 值占了前 520 名，并列极密）。但这仍**不是我们独有的偏差**——
SGLang 自己从 L2 起就不可重复（§12.28），而我们的 0.0206 对两套参考完全相同，说明它是一个固定的
系统性量而非随参考选择抖动。

### 12.36 L1 的 y 差 5.6 ulp（继承 L0），以及一个诊断工具的坑：`attn{L}_x.bin` 对 L>0 不可信

`attn_y_audit.py`（新工具：先验证 mHC 的 post 在 dump 上可逆，再反解 y）在两层上的判决：

| 层 | 用哪一流反解 | post | 反解闭合 | our y vs y_ref | cos | 逐 bit | |d| 的 ulp 分布 |
|---|---|---|---|---|---|---|---|
| L0 | stream 3 | 0.136196 | 100.00% | 4.984e-03 | 0.9999876 | 0/5120 | 0ulp:1685 1ulp:1834 2ulp:615 3ulp:272 |
| L1 | stream 0 | 0.191603 | 100.00% | 2.179e-02 | 0.9997641 | 0/5120 | 0ulp:368 1ulp:715 2ulp:606 3ulp:529 |

两层的 `identity check（ref coef + ref y -> R'）` 都是 **0.000e+00（逐 bit）**，所以反解是合法的。

L1 的 5.6 ulp 是**继承**的：L0 的 y 差 1.3 ulp → L0 的 MoE → L0 边界残差
（我们的 `moe0_streams_eg` vs 参考 `L01_eg_y` = **5.699e-03**，逐流 9.5e-3/3.6e-3/1.8e-3/5.4e-3）
→ L1 的 x 差 ~5.7e-3 → L1 的 y 差 2.18e-2。**不是 L1 自己的新错误。**

#### 一个**已被自己排除**的假警报：`attn1_x` 不能拿 attention 的 triplet 去解释

一开始用参考的 `L01_ct_pre` 去崩缩参考自己的 `L01_eg_y`，算 `norm(Σ pre_c R_c)·γ`，与 `attn1_x` 差 **24.2%**
（cos 0.9704，只 38/5120 逐 bit）；用我们自己的残差和我们自己的 pre 也一样。看着像链的 x 接错了。

**不是。** 两点一查就清楚：

    attn1_x == moe0_x_out ?  5120/5120     relRMS 0.000e+00
    attn1_x == attn0_x    ?    12/5120     relRMS 8.076e-01      <- 那条"读到 embedding"的老 bug 确实已经不在

`attn{L}_x` 对 L>0 就是**上一个子层**的 `x_out`（`full_decode.py` 里 `x = [wg["x_out"] for wg in ws]`），
而 L1 attention 的输入是 **L0 的 MoE** 的 post 输出 —— 那个 post 用的是 **MoE 子层自己的 triplet**，
不是 attention 的。dump 只记录 attention triplet（`L{L}_ct_*`），所以拿 `L01_ct_pre` 去解释当然差 24%。

自洽性本身还有一条独立证据：`norm(pre . moe0_streams_eg) * layers.1.attn_norm.weight` 与 `attn1_x` 差 2.4e-1，
而换成 `layers.1.ffn_norm.weight` / `layers.0.ffn_norm.weight` 差 0.84/0.85 —— 只有 `attn_norm(L1)` 量级对，
这正是 `mhc_into(..., "attn", ffn_norm)` 那条"post 负责算下一个子层的输入"的接线。

→ 结论：**dump 是对的，链是自洽的**；只是 L>0 的 attention 输入**无法**从 dump 里的 attention triplet 复算。
要复算必须拿 MoE 的 triplet，而它没有被记录。

---

## 12.37 SGLang decode 真正跑的注意力核找到了，而且能离线调用 —— L0 的种子不在注意力核

**这一节把 12.35 的"1.3 ulp 无从下手"推进到可执行的层面。**

### 12.37.1 排查路径：三条都不是，最后一条是

`deepseek_v4_backend.py` 的 decode 分派实际有三条，我逐个核对过前提：

| 分支 | 条件 | 我们的配置 | 结论 |
|---|---|---|---|
| `swapab_attention` | `can_use_swapab_attention(..., layer.tp_q_head_num, ...)`，要求 `num_heads ∈ (16, 32)` | `n_local_heads = 64 // attn_tp_size = 8` | **不可用**（12.19 那次分析的假设错了） |
| `_forward_mega_decode` | `cos_sin_cache is not None`，即 `use_mega_attention` | `dsv4_attn_backend == "auto"` 且 attention TP=8 > 1 → 解析成 `"flashmla"` | **不是它** |
| `flash_mla_with_kvcache` | 其余 | 命中 | **是它** |

`'auto'` 的解析规则就在 `arg_groups/deepseek_v4_hook.py:_default_dsv4_attention_backend`：

    # MegaAttention pads each rank to a 64-head tile under attention TP.
    or (cfg.disaggregation_mode == "null" and _attention_is_tensor_parallel(cfg))
    → return "flashmla"

所以 decode attention = **FlashMLA 的预编译 wheel**（`sgl-flash-mla`，`import flash_mla`），
`use_legacy = not swa_key_layout().is_interleaved` —— 我们是 `v41_new`（addr dump 的 `pool.kv_layout`），
所以 `use_legacy=False`，走 2026.09.30 之后的新 kernel。

**8 个 head 为什么能跑？** 不是 head 复制，是 pad：

    n_local_heads = 8            # 真 head
    kernel_num_heads = 64        # deepseek_v4.py:1251  "64 if self.n_local_heads <= 64 else self.n_heads"
    qt = new_zeros(1, 1, 64, 512); qt[0, 0, :8] = q8

实测：喂 8 head 直接 `RuntimeError: sparse_decode_fwd ... Unsupported h_q: 8`；零填到 64 head 就正常。
（`pack_query` 那套 [T,32,64,16] 的转置只给 MegaAttention 用，普通 flash_mla 这条不用它。）

### 12.37.2 离线 harness：`l0_flashmla.py`

用参考自己的状态喂真 kernel，唯一要自己补的是当前 token 的 K（decode 那一步才写，dump 里那一槽**全零**）：

    pool    /tmp/dec_coef8/r00_L00_window_cache.bin   [133, 256, 1, 528] uint8 (page 256, 528 B/token, v41_new)
    loc     meta_addr_rank00.json: loc = pos + 256；decode token 的 loc = 32162 = page 125 slot 162
    indices swa_page_indices 是**最新在前**：32162, 32161, ..., 32035
    q       bf16(rope(bf16(Σ wq_b parts)))
    sink    layers.0.attn.attn_sink[r*8:(r+1)*8]，零填到 64

### 12.37.3 结果：注意力核与我们只差 0.08 bf16 ulp

64-head 零填、`use_legacy=0`、128 个 index 全给（无 split-KV，topk 128 < SPLIT_KV_MIN_TOPK=641）。
把返回的 `[1,1,64,512]` 取前 8 个 head，再做逆 rope 与我们 `attn0_r*_o_q.bin` 比：

| rank | noPE 逐 bit | relRMS | cos |
|---|---|---|---|
| 0 | 3385/3584 | 3.500e-04 | 0.999999939 |
| 1 | 3243/3584 | 1.006e-03 | 0.999999502 |
| 2 | 3285/3584 | 5.988e-04 | 0.999999821 |
| 3 | 3075/3584 | 1.069e-03 | 0.999999432 |
| 4 | 3400/3584 | 3.100e-04 | 0.999999952 |
| 5 | 3361/3584 | 3.569e-04 | 0.999999936 |
| 6 | 3215/3584 | 6.815e-04 | 0.999999768 |
| 7 | 3252/3584 | 6.414e-04 | 0.999999796 |

86–95% 的值逐 bit 相同，相对差 0.08–0.27 bf16 ulp。（剩下的差里有一部分必然来自我自己补的那行 K：
SGLang 的 `fused_norm_rope_attn_rope_cast` 把 norm+rope+quant 融成一个 kernel，我的 528 字节是照 `kv_post.cuh`
自己拼的，无法与真值对照。）

**对准 fp64 真值看谁更准**（128 行 K、我们的 q、fp64 softmax、sink 用 `1/(1+exp(sink-lse))`）：

    head    ours-fp64    fml-fp64    ours-fml
       0   4.814e-03   4.814e-03   3.021e-04
       1   3.214e-03   3.213e-03   1.206e-04
       2   1.233e-03   1.272e-03   3.504e-04
       4   2.869e-03   2.876e-03   7.073e-04
       6   4.685e-03   4.685e-03   1.782e-04

两者离 fp64 真值的距离**几乎完全一样**（差异都被 bf16 输出舍入主导），彼此只差 0.08 ulp。
→ **注意力核不是 L0 那 1.3 ulp 的主种子**，想让 o 逐 bit 相同除非用他们那个 kernel。

### 12.37.4 归因：同一个 wo 代码下换 o，gap 从 4.10e-3 掉到 2.27e-3

把两套 o（我们的 / flash_mla 的）灌进**完全相同**的参考 wo 路径（wo_a → 输出舍 bf16 → act_quant → wo_b，8 个 group 求和）：

    our o_q      wo_a->bf16: relRMS 4.102e-03  cos 0.9999916018
    flash_mla o  wo_a->bf16: relRMS 2.274e-03  cos 0.9999974146
    两套 o 走同一 wo 代码互差: relRMS 3.382e-03  cos 0.9999942857

所以：注意力核的 0.08 ulp 经 wo_a 的 e4m3 量化被放大到 ~3.4e-3，占掉 y 那 4.10e-3 里的一大半；
**剩下的 2.27e-3 = 0.58 个 y 的 bf16 ulp，是 wo 路径自身的算术（两个 GEMM 的归约顺序 + 量化边界）**。

顺带确认了 wo 路径的结构在两边是**一致的**（不是这里接错）：

- 我们：`o(bf16) → wo_a(e4m3×e8m0) → fp32 → act_quant e4m3/32-block ue8m0 → wo_b`
- SGLang：`deepseek_v4.py:3120` 起，`n_local_groups=1` 使 `can_use_fused_wo_a(1,1024)==False`（要求 groups ∈ (2,4)），
  于是走 `_FP8_WO_A_UE8M0` 分支：`sglang_per_token_group_quant_fp8_dsv4_wo_a_with_rope`（e4m3、group 128、ue8m0，
  **逆 rope 融在量化 kernel 里**）→ a8w8 mxscale GEMM → 输出舍 bf16 → wo_b
- 量化粒度实测也确实对得上：我们的 `oa_q` 用 `zq_split` 解出来是 **32 个 ue8m0 scale / 1024 列 = block 32**，
  与 mxfp8 一致；把 block 在 16/32/128 之间扫，y 的 relRMS 四位有效数字不变。

### 12.37.5 下一步（可按顺序推进）

1. `wo` 路径那 0.58 ulp：把 `sglang_per_token_group_quant_fp8_dsv4_wo_a` 和他们的 mxscale GEMM 也离线调起来
   （1 张卡就够，不需要 8 卡引擎），逐段替换我手写的 numpy，看 2.27e-3 掉到哪一步。
2. 我补的那行 K 需要验证：把参考 dump 的 `set_swa_key_buffer_radix_fused_norm_rope` 的产物拿来
   （即某个已写位置的 528 字节）与我们独立复算的同位置 K 对比；这需要在参考侧 dump 一行 K。
3. 权威 dump（SGLang 侧直接记录 `o` / wo_a 输出）**当前被别的作业占满 8 张卡挡住**，
   代码侧已经想好怎么加（wrap `MqaAttentionBase._project_wo_b` 记 wo_a 输出、wrap
   `DeepseekV4AttnBackend.forward` 记 rope 空间的 o），等卡空了跑一次就是全 40 层的真值。

**顺带排除的两个假设**（都是便宜的全局扫描，结论是"我们本来就对"）：

- ue8m0 的 scale 规则：`ceil(log2(amax/448))` 是对的。换成 round-to-nearest 会让 y 的 relRMS 从
  4.10e-3 恶化到 3.79e-2（block 32），floor 更差到 1.41e-1。所以 `round_sf=True` 不是"改 quant 规则"的意思。
- wo_b 输入量化的 block：16/32/128 扫过去 y 的 relRMS 四位有效数字**完全不变**，这条不敏感，不必再纠结。

---

## 12.38 40 层对齐的真实记分板：比较工具错了三处，参考本身从 L2 起就没有唯一答案

**这一节纠正了"40 层在 L≥2 崩塌"的结论。链没崩，是 `cmp_all.py` 在拿错位的参考量做比较。**

§12.13 已经确立了"参考从 L2 起不确定、每层 <1% 不可达"这条事实（那里用的是 `cmp_sgl.py` 和 `ref_spread.py`，
两个工具本来是对的）。这一节补的是：**粗粒度的 `cmp_all.py` 一直报 0.6–1.6 的假误差**，原因是三处映射错误，
修好之后三个工具完全一致；另外把"链是否落在带内"变成一条命令（`cmp_band.py`）。

### 12.38.1 三处比较错误（都用实测定的，不是推的）

**(1) MoE 残差要和 `L{L+1}_mpost` 比，不是 `L{L}_mpost`。**
参考的层边界是融合的：layer L-1 的 FFN post 在 layer L 的前向里算，所以记在 L 的 id 下。实测（`moe{L}_streams_out`）：

    L     vs L{L}_mpost    vs L{L+1}_mpost
    2       7.05e-01          2.36e-02
    3       5.53e-01          2.75e-02
    5       5.12e-01          3.11e-02
    10      4.32e-01          7.68e-02
    20      4.08e-01          1.09e-01

**(2) 拿 `moe{L}_streams_eg` 去比也是错的。** engram 在 post *之后*改写残差，而参考记的是 engram 之前的状态。
所以两边都用 `streams_out`。这条同时给了一个反解闭合的判据：链的 post 在 `moe{L-1}_streams_out` 上闭合到
1.3e-4–5.0e-4，换 `_eg` 就是 5.4e-1（L1），换 `attn{L-1}` 是 9e-2–1.7e-1 —— 这就是 post 真正用的 R。

**(3) `attn{L}_x_out` 不能直接比。** 它是 rank 0 的 o_proj **all-reduce 前的部分和**（L0 norm 8.9 vs 参考 35.6），
直接比会给出 1.2 的假误差。必须用 `infer_s` 反解。另外 `_mpost_x` 也不可靠：参考的 MoE 是
`leave_output_sum=True`，交给 post 的是**未合并的 pieces**，`_as_tensor` 只会挑其中一个。

顺带确认参考内部的自洽性（这也是理解层边界的钥匙）：

    L02_apost  vs L03_in   : relRMS 0.000e+00   20480/20480 逐 bit
    L02_apost  vs L02_mpost: relRMS 4.50e-01
    L02_apost  vs L03_st   : relRMS 6.62e-01

即层启动那一刻残差里仍是**上一层 attention 的 post**，MoE 的写回发生得更晚。

### 12.38.2 参考自己的散布带：这就是唯一诚实的标尺

同一 checkpoint、同一 token、同一设置，**四次独立启动**（dec_coef2/6/7/8）：

| 层 | `_apost` (20480) | `_apost_x` (5120) |
|---|---|---|
| L0, L1 | 0.0（逐 bit） | 0.0 |
| L2 | 6.35e-03 | 1.29e-02 |
| L5 | 2.51e-02 | 3.64e-02 |
| L10 | 8.01e-02 | 7.98e-02 |
| L20 | 1.29e-01 | 1.16e-01 |
| L39 | 1.05e-01 | 7.20e-02 |

**所以目标里"40 层全部 relRMS 远小于 1%"物理上达不到**：L10 之后参考自己四次运行就差 8–13%，没有一个"正确答案"可供对齐。
能问的只有：我们的链落不落在这条带里。这不是弱判据——错误的 kernel 会远远跌出带外（MoE 在修好映射前是 0.6–1.6），
正确的会落进来。

### 12.38.3 记分板（`cmp_band.py`，3 次参考运行自建带，`--slack 1.5`）

    chain            inside   outside   最差比率
    chain_idx        101/117     16      L2 apost 3.62
    chain_m3         101/117     16      同上（完全一致）
    chain_m10        101/117     16      同上

逐层：

| 层 | 量 | 带 | 我们 | 比率 |
|---|---|---|---|---|
| 0 | apost | 0（逐 bit） | 2.647e-03 | 参考无噪声，我们差 0.26% |
| 0 | apost_x | 0 | 4.984e-03 | 0.50% |
| 1 | apost | 0 | 9.186e-03 | 0.92% |
| 1 | apost_x | 0 | 1.917e-02 | 1.92% |
| 1 | mpost | 0 | 1.394e-02 | 1.39% |
| 2 | apost | 5.69e-03 | 2.059e-02 | **3.62 带外** |
| 2 | apost_x | 1.29e-02 | 4.041e-02 | **3.13 带外** |
| 2 | mpost | 1.04e-02 | 2.362e-02 | **2.26 带外** |
| 3 | apost | 1.66e-02 | 2.673e-02 | 1.61 带外 |
| 3 | apost_x/mpost | — | — | 1.13 / 1.26 带内 |
| 4–39 | 全部 | — | — | **0.80–0.91 带内** |

**结论**：L4 到 L39 的 111 个量全部落在参考自己的带内，比率稳定在 0.85 左右（比参考自己两次运行之间还接近）。
"带外"的 16 个里 5 个是 L0/L1（参考在那里是逐 bit 确定的，没有带可以容纳误差），3 个是 L2，1 个是 L3 的 apost。

### 12.38.4 剩下的实质工作就是 L0/L1 的最后一位

L0 的 0.26%/0.50%、L1 的 0.92%/1.92% 是**唯一真实的误差**，L2 的 2.06% 是它经 mHC post 敏感放大后的结果
（post 系数在 L2 是 0.219，comb 主导，1 ulp 的 y 就会在残差里放大）。这条链是：

    L0 y 差 0.50%（≈1.3 个 bf16 ulp，§12.35/§12.37）
      -> L0 残差 0.26%
      -> L1 输入 0.57%
      -> L1 y 差 1.92%
      -> L2 2.06%（参考带 0.57%）
      -> 从这里开始参考自己也开始发散，带迅速变宽，我们在 L4 被"追上"

所以要把 L2 拉进带内，唯一的路还是 §12.37 那条：让 L0 的 y 从 1.3 ulp 降到 0。注意力核已经不是种子了
（与 flash_mla 只差 0.08 ulp），剩下的是 wo 路径那 0.58 ulp。

### 12.38.5 端到端：我们的链是确定性的，但缺参考 logits

两次独立运行（`chain_ok` 10:38、`chain_idx` 13:01）的 logits：

    chain_ok vs chain_idx row0: relRMS 0.000e+00  （逐 bit 相同）
    top5 [687, 1529, 32125, 29335, 5183]  = [23.125, 22.0, 20.75, 20.75, 20.625]

我们的链完全可复现（比参考更强）。但**参考侧从未 dump 过 decode 步的 logits**，所以"最终 logits 对齐"这一条
目前无法验证。要验证得在 `dump_hook.py` 里加一条 logits 记录（和 §12.37.5 的 wo_a/o 记录一起，一次运行就能拿到），
而那需要一个空的 8 卡窗口。

### 12.38.6 工具改动

- `cmp_all.py`：MoE 残差改用 `L{L+1}_mpost`；`attn{L}_streams_out` 对 `L{L}_apost`；MoE 的 out 反解禁用（`_mpost_x` 是 pieces）；反解的 R_in 只用 `streams_out`。
- `cmp_band.py`（新增）：用 3 次参考运行自建每层每量的带，输出 `relRMS(ours, ref) / band`，`ratio <= slack` 判带内；逐 bit 层单独报告。

### 12.39 wo 路径复刻到了哪一步（r9）

**(1) TP=8 时 wo_a 的快路径全部不触发。** `_apply_wo_a_bf16_matmul` 的每个专用分支都要求
`o.shape[1:] == (2, 4096)`，即 `n_local_groups == 2`；TP=8 是 `n_local_groups = 1`（o_groups 8 / TP 8），
所以 decode 的 GEMV、small-batch、bmm 全部跳过，落到最后一行：

    return torch.einsum("tgd,grd->tgr", o, wo_a)

**(2) `wo_a_fp8 = False`**：`wo_a_fp8_gemm_enabled` 在 CUDA 上要求 `weight_block_size == [128,128]`，而 V4.1 的
checkpoint 是 `[32,32]`，所以 wo_a 在加载时**反量化成 bf16**，走上面的 bf16 einsum，不是 DeepGEMM 的 fp8_einsum
（那条路按 `block == [32,32] and _is_hip` 只在 ROCm 上开）。之前我按 fp8 einsum 推的那条路径在 CUDA 上不成立。

**(3) bf16 权重是无损的**（e4m3 × 2^k 只有 4 位有效位，bf16 有 8 位），所以 `wo_a -> bf16` 的舍入对结果**零影响**：
`wo_a->bf16 0/1` 与 `out->bf16 0/1` 四种组合里，决定性的是**输出**那个 bf16 舍入（9.56e-3 -> 4.10e-3），不是权重。

**(4) `torch.einsum` 换成我手写的 fp32 matvec 不改变 y**：两条路给出 4.1016e-03 一模一样，所以 wo_a 这一级的
算术已经复刻到位。

**(5) 跨 rank 的 bf16 部分和是真的**。wo_b 是 RowParallelLinear（tp=8，每 rank 1024 列），先出 bf16 partial 再
all-reduce。把每个 partial 先舍到 bf16 再求和：

    flash_mla 的 o : 2.274e-03  ->  1.712e-03   （0.44 ulp，变好）
    我们的 o       : 4.102e-03  ->  4.305e-03   （变差）

**结论**：剩下的 0.44–0.58 ulp 不是某一行代码写错，而是 (a) 我们的 `o` 与 SGLang 真正的 `o` 还有 ~0.08 ulp 的差，
(b) deep_gemm/flashinfer 的 block-scaled GEMM 的 k 序累加，(c) NCCL all-reduce 的归约树。三者都无法在不知道参考
侧 `o` 的情况下分离——这就是 r6 那个被 8 卡占用挡住的 dump 的价值：拿到 SGLang 自己的 `o` 和 wo_a 输出，才能判断
这 0.5 ulp 该记在谁账上。在此之前，L0 的 0.50% 已经是"最后一个 bf16 位"级别的差，而不是结构性问题。

**交叉验证**：修好后的 `cmp_all.py` 与 `cmp_sgl.py` 在 `/tmp/chain_idx` 上逐行一致 ——

    cmp_all  L36 moe 0.0862 / attn37 0.0877 / attn38 0.0919 / attn39 0.0884
    cmp_sgl  L36 moe 0.086  / attn37 0.088  / attn38 0.092  / attn39 0.088
    cmp_sgl  最差子层 L18 moe cos +0.9913

三个工具（`cmp_all.py` / `cmp_sgl.py` / `cmp_band.py`）现在对同一份 dump 给出同一组数。
`cmp_band.py` 的汇总：117 行里 101 行带内，带外 16 行 = L0/L1 的 5 行（参考逐 bit，无带可容）+ L2 的 3 行 + L3 的 apost
+ L3–L4 的少量行。

## 12.40 r6 完成：拿到了参考侧的 o / oa / y_part / logits，L0 的最后一位归因到底（2026-11-27 15:30）

四次 dump 尝试里有三次被同一个东西打死，原因不在记录逻辑，而在**记录逻辑之前**：

    dump_hook.py `_layer_forward` 第一行就是 `dec0, last0 = _is_decode(fb), _last_pos(kw, args)`，
    `_last_pos` 做 `int(p.max())` —— 一次 device 同步。而 prefill runner 用 breakable CUDA graph 捕获
    整段 prefill（`disable_cuda_graph=True` 只关 decode 图，不关 prefill 图），同步在捕获区内非法：

        CUDA error: operation failed due to a previous error during capture
        Fail when using backend: breakable for prefill runner.   -> 8 个 rank 全挂，prompt 都没喂进去

修法是三件事，缺一不可：`_capturing()` 提到读取张量之前；`dump_prefill.py` 加
`disable_prefill_cuda_graph=True`（`cuda_graph_backend_prefill="disabled"` 单独用不够，prefill 仍会捕获）；
`_last_pos` 增加 `fb.positions` 回退（这个 tree 的 positions 挂在 ForwardBatch 上，不在 layer 的 kw/args 里，
没有这条 prefill 层行一个都写不出来，pools 却照写 —— 只有"layers 目录是空的"这一个症状）。

另外两个静默失败：`mhc.AttnOutput` 是 NamedTuple，字段叫 `partial` 不是 `data`；logits 的字段叫
`next_token_logits` 不是 `logits`。两者都不抛异常，只是什么都不写。

还有一个**记错步**的坑：decode 步必须按 **position** 认，不能按"第一个 decode forward"认。引擎会先跑一个
1-token 的 dummy decode，它的入 token 是 103084 —— 实测 `L00_in` 与 `embed[103084]` cos = 1.0，与
`embed[4588]` cos = 0.04。那种错看起来像"注意力核写错了"，其实是记错了步。现在按 `_AT + 1`（=31906）门控，
并且 `--max-new 2` 保证真实 decode 一定发生（`[4588, 343]`）。

### 12.40.1 参考侧现在有什么

    /tmp/dec_io2/layers/step1/            L00..L39 × 11 个量 = 440 行  + logits.f32（129280，全词表）
    /tmp/dec_io2/layers/step1/io/         40 层 × 8 rank × 4 个量 = 1280 个 bf16 文件
                                          o_rope（核返回的 rope 空间 o）o_inv（wo_a 的输入）oa（wo_b 的输入）y_part（未归约的 bf16 部分和）

`L00_in/L00_apost/L00_apost_x` 与 `dec_coef8` **逐位相同**（20480/20480、5120/5120），所以这趟记的确实是那一步。

### 12.40.2 L0 的最后一位：注意力核 0.16 ulp + wo 尾段 0.43 ulp

**第一段 —— 我们的 wo_a GEMM 用参考自己的 o 喂进去，与参考自己的 oa 比：**

    rank   raw fp32 relRMS   bf16 舍入后      逐 bit 命中
    r0       1.663e-03        0.000e+00      1024/1024
    r1       1.709e-03        3.806e-05      1022/1024
    r2       1.630e-03        3.705e-04      1018/1024
    r3       1.731e-03        0.000e+00      1024/1024
    r4       1.665e-03        0.000e+00      1024/1024
    r5       1.660e-03        0.000e+00      1024/1024
    r6       1.682e-03        7.246e-07      1023/1024
    r7       1.708e-03        1.830e-04      1020/1024

8185/8192 逐位相同。**wo_a 这一级没有可归因的误差**（raw fp32 的 1.66e-3 全部落在 bf16 舍入之内）。

**第二段 —— 参考自己的 oa 喂进我们的 wo_b + bf16 partial + 求和：**

    y 的 relRMS 1.6667e-03   （= 0.43 个 y 的 bf16 ulp）   cos 0.9999986130

这一段是纯实现序（wo_b 的 k 序累加 + bf16 partial + all-reduce 树），换注意力核无法消除。

**第三段 —— 换成我们自己的 o，同一条代码：**

    SGLang 的 o -> 我们的 wo : y relRMS 1.6667e-03
    我们的 的 o -> 我们的 wo : y relRMS 4.3052e-03
    -> 注意力核贡献（平方差）  3.97e-03        wo 尾段 1.67e-03

而我们的 `o_q` 与参考 `o_inv` 的直接差是 **6.266e-04（0.16 bf16 ulp），中位 0 ulp，91.6% 逐 bit 相同**
（30007/32768）：

    r0 3.500e-04 94.6% | r1 1.005e-03 90.4% | r2 5.988e-04 91.9% | r3 1.069e-03 86.1%
    r4 3.100e-04 95.1% | r5 3.569e-04 94.0% | r6 6.816e-04 89.7% | r7 6.414e-04 90.8%

**所以 r10 的答案是：两边都有，注意力核是主项。** o 上 0.16 ulp 被 wo_a -> bf16 -> act_quant（32 元素块的
block max，非线性量化器）放大 6.3 倍成 1.05 ulp 的 y —— 8.4% 的元素差最后一位，就会翻转约 8% 的量化/舍入判决。
0.43 ulp 那段（wo_b 的 k 序 + partial + all-reduce）是任何复刻都消不掉的噪声底。

### 12.40.3 端到端：40 层链的 logits 与参考同 argmax、同 top-5 集合

    /tmp/dec_io2/layers/step1/logits.f32（参考，decode 步 → 预测 343）
    /tmp/chain_ok/logits.bin 与 /tmp/chain_idx/logits.bin 逐 bit 相同（我们可复现）

    relRMS 3.9107e-02   max|d| 1.312   （logits 绝对值 13–23）

    ref   top5 [343, 764, 16062, 2591, 14667]
    chain top5 [343, 764, 16062, 14667, 2591]

argmax 一致（343，正是 SGLang 这一步生成的 token），top-5 是同一个集合，只是第 4/5 名互换。
3.9e-2 是 40 层每层 ~1 ulp 累积后的 logits 相对差，不是新的结构性问题。

### 12.40.4 参考自带的散度（再次确认，这次用同 hook 的两次运行）

`/tmp/dec_io1` vs `/tmp/dec_io2`（同 prompt 同权重同代码，两个独立进程）：

    step1 的 440 个 f32 里，逐 bit 相同的只有 29 个（全部属于 L0/L1）
    L02_apost     relRMS 5.427e-03
    L03_apost     relRMS 1.566e-02
    L03_mpost     relRMS 9.890e-03

即 stock SGLang 自己在 L2 的散度（5.4e-3）就已经**大于**我们 L0 的差（4.3e-3）。L0/L1 的带是 0（两次运行逐 bit
相同），所以 L0 那 1 ulp 是必然"带外"的，但它比参考自己在 L3 的噪声还小。

### 12.40.5 logits 也进了参考自己的带（0.87x）

第三趟 `/tmp/dec_io3`（同 hook 同参数）给出第二个参考 logits，于是可以量出 logits 的带：

    参考自身两次运行       relRMS 4.4930e-02   max|d| 1.250
    我们的链 vs dec_io2    relRMS 3.9107e-02   max|d| 1.312
    我们的链 vs dec_io3    relRMS 4.0904e-02   max|d| 1.246
    -> 带比 0.87x（带内）

    ref2  top5 [343, 764, 16062, 2591, 14667]
    ref3  top5 [343, 764, 16062, 14667, 37481]     <- 参考自己第 5 名都换了
    chain top5 [343, 764, 16062, 14667, 2591]

三次运行 top-3 完全一致（343、764、16062），前两名的差值都在 1.25–1.5 之内；参考自己第 4/5 名都在跳
（2591 <-> 37481）。**我们的链在 logits 层面不比 SGLang 自己两次运行之间更远**，这是"端到端对齐"能被检验的
最强形式 —— 逐 bit 相等在这套推理栈上对参考自己都不成立。

### 12.40.6 结论

| 层面 | 我们的链 vs 参考 | 参考自己的带 | 比 |
|---|---|---|---|
| L0 的 y | 4.31e-03（1.1 ulp；注意力核 0.16 ulp + wo 尾 0.43 ulp） | 0（逐 bit） | 带外，但比参考 L3 的自噪声（1.6e-2）还小 |
| L4–L39 的 attn/moe | — | — | 0.80–0.91x，带内（`cmp_band.py` 101/117 行） |
| logits | 3.91e-02 | 4.49e-02 | **0.87x，带内**，argmax 一致 |

"40 层逐层 relRMS << 1%" 这个要求在 stock SGLang 上物理不可达（它自己两次运行在 L3 就散到 1.6%），
可检验的目标只能是"不比参考自己两次运行之间更远"，而这个目标目前三层（逐层、logits、top-k）都达到了。

## 12.41 r11 结案：那 0.16 ulp 不是缺陷，是两种同等精度的归约序之差（2026-11-27 23:50）

r11 原本的目的是"对齐 flash_mla 的 PV/softmax 归约序，把 L0 从 1.1 ulp 压到 0.5 ulp"。动手前先问了一个前置问题：
**这 0.16 ulp 到底是"我们更差"，还是"两边都是同一个精度等级里的两个样本"？** 用 fp64 真值（在同样的 bf16 q、
bf16 kv 行、fp32 sink 上做精确 softmax）判定，答案是后者。

### 12.41.1 两边到真值的距离完全相同

    ours vs fp64 truth     relRMS 7.277e-03   median 0.93 ulp   |x-truth| 总和 1.590158e+01
    reference vs truth     relRMS 7.277e-03   median 0.93 ulp   |x-truth| 总和 1.591998e+01
    -> 比值 0.999

（8 个 rank 单独看也一样：0.99–1.00x。）也就是说我们的 `o` 并不比 SGLang 的差，反过来也不更好 —— 两者都是
"P 舍到 bf16 + MMA 内部累加序"这一档的产物，距离真值中位数 0.93 ulp。

在 8.4% 有差异的元素里，我们更接近真值的是 4.3/8.4 ≈ **51%** —— 抛硬币。

### 12.41.2 也不是"我们偏离了 checkpoint 的参考实现"

checkpoint 自带 `inference/kernel.py sparse_attn_kernel`（两块 online softmax，P 每块舍 bf16）。用 numpy 复刻它：

    ckpt-replay vs OURS   0.006593
    ckpt-replay vs REF    0.006580
    -> 我们的 o 相对参考实现的距离是 SGLang 的 1.00x（8 个 rank 分别 0.98–1.00x）

numpy 复刻本身也到不了任何一方的精度（MMA 的累加序不可复现），所以这条比较只能说明"双方同等贴合参考实现"。

### 12.41.3 差异是纯噪声，没有系统性偏差

    逐元素：+48.6% / -51.4%（系统性偏差会是 ~100:0）
    带符号均值 -3.666e-06  vs 平均 |d| 1.751e-04  -> 偏差/|d| = -0.021
    差的总和 -1.012e-02（互相抵消，不是整体偏移）

    列分块（noPE 四段 + RoPE 一段）：7.5% / 8.5% / 9.0% / 9.3% / 7.4%
    各 rank：4.9%–13.9%（每 rank 只有 4096 个元素，标准差 0.43%）
    各 head（rank0）：1.2%–11.5%

均匀分布、符号对半、无 offset —— 这是"同一个数学的两种累加序"，不是某一段代码写错。

### 12.41.4 结论：不做 bit-mimicry

要把 `o` 做到逐位相同，只能复刻 flash_mla 的 MMA/softmax 指令序，而这**不带来任何精度收益**（0.999 比值），
只是把一个随机样本换成另一个随机样本。收益上界也很小：y 现在是 1.05 ulp，即使 `o` 完全逐位相同，剩下 wo_b
尾段那 0.43 ulp 仍然在（§12.40.2 第二段），所以最好情况是 1.05 -> 0.43 ulp。

而端到端已经证明这点误差被参考自身的噪声完全吸收：logits 3.91e-02 = 参考自身带宽的 **0.87x**（§12.40.5）。

**r11 结案：不是缺陷，不动作。** 若将来出于"逐位可复现某个特定引擎"的目标确实需要，唯一可行的路径是在静态核里
按 flash_mla 的 tile/累加序重写 PV，而不是调参数逼近。

## 12.42 decode TPOT 的归因：92% 是每次 launch 的固定开销，不是 32K 的计算（2026-11-27）

目标从"对齐"转到"TPOT"，所以第一件事是把 32K 一步的时间**拆开**，而不是先猜优化点。全部数字都是量出来的。

### 12.42.1 上下文长度几乎不影响一步的时间

同一条链（swa/adam 与 21 层前缀），只改 `--pos`：

| 位置 | 6 次 launch 的 span 和 | 42 次 launch 的 span 和 |
|---|---|---|
| 1024 | 296.6 us | 2208.4 us |
| 8192 | 309.1 us | — |
| 31906 | 338.0 us | 2376.6 us |

**31 倍上下文，42 次 launch 只多 7.6%**（2208 -> 2377 us）。逐层看更清楚：

| 变体 | 位置 1024 | 位置 31906 | 结论 |
|---|---|---|---|
| `swa`（attn0/1） | 44.5 / 45.1 us | 44.6 / 43.8 us | **与上下文无关** |
| `full_r2`（attn2/8/14） | 68.0 / 65.6 / 68.3 us | 94.3 / 94.8 / 94.4 us | +26 us，**唯一随 32K 线性增长的部分** |
| `reuse`（其余 30 层） | ~48-50 us | ~48-51 us | 与上下文无关 |
| MoE（全部 40 层） | ~48-53 us | ~47-53 us | 与上下文无关 |

所以 32K 真正多付出的只有 8 个 indexer 层（3 个 `full` + 5 个 `reindex`），合计约 400 us；
其余 ~3.7 ms 是**每次 launch 的固定开销**，在 1K 上一样要付。

这条结论直接改写了优化方向：把 32K 的 attention 算快**没有用**（它只占 8%），
必须减少 launch 次数、或压低每次 launch 的固定开销。

### 12.42.2 固定开销的构成（内核自带硬件 stamps）

`full_decode.py --probe` 用 `stamp_start / start_barrier / stamp_end` 分期读出来（无需 profiler）：

```
probe attn0 (swa, pos 31906): span 42.4 us over 148 SMs
  SMs enter the kernel over 224 us (148/148 stamped)
  all SMs past the start barrier at +64.2 us
  back-to-back 30x: 345.6 us a launch (device wall), of which span 42.0 us
```

- `span_us()` 的口径是"最后一个 SM 过起始栅栏 -> 最后一个 SM 结束"，**不含入核与栅栏**，
  所以把 80 个 span 相加（4.14 ms）会**低估**一步的真实代价；8 进程服务形态实测是 8.16 ms（§12.42.4）。
- 单进程 8 卡跑 back-to-back 是 345.6 us/次，但那是 host-reset 构建（每次 launch 前 host memset 全部 buffer），
  **不代表服务路径**，只说明这条测量通道不可用于 TPOT。

### 12.42.3 每任务成本：单 token GEMV 慢在依赖跳数，不在带宽

timing build 的 per-node 汇总（attn0，32K，busy = 该 node 全部任务的耗时之和）：

| node | busy | 任务数 | 单任务 |
|---|---|---|---|
| `wo_b_part` | 1919.5 us | 80 | 24.0 us |
| `y_sent` | 1159.2 us | 40 | 29.0 us |
| `oa_q` | 211.4 us | 8 | 26.4 us |
| `wo_a_part` | 410.1 us | 16 | 25.6 us |
| `o_q` | 164.5 us | 8 | 20.6 us |
| `wq_b_part` | 301.1 us | 32 | 9.4 us |
| `wq_a_part` | 192.5 us | 40 | 4.8 us |

T=1 的一次 GEMV 读的权重约 5 MB，满带宽下应 ≈0.6 us，实测 20-29 us —— 差 30-50 倍。
所以瓶颈是**跨 SM/跨 rank 的依赖跳数（计数器轮询）**，不是 HBM。一层约 8 跳，每跳几 us，凑成 ~44 us 的地板。

### 12.42.4 正向对照失败：`TPB=1` 并不干净（§8.1 原来"待补跑"的那一格）

§8.1 的真值表里 `TPB=1`（内核自带栅栏）一行此前是**待 GPU 空闲后补跑**。今天跑了，结果是：

```
DSV41_STATIC_TPB=1（默认），e2e_decode.py --layers 6
[e2e] NOT FINITE in 6 sublayers, the first is attn L3: attn L3, moe L3, attn L4, moe L4, attn L5, moe L5
```

- 6 层就确定性复现，首个非有限仍是 **attn L3（第一个 `reuse` 层）**，与 `tpb0_nosync` 的位置一模一样。
- 确认 `-DSTATIC_TP_BARRIER` **确实编进了** nvcc 命令行（`e2e_t_tpb1.log` 里 96 次出现），
  所以不是"flag 没进去"，是**栅栏没能给出 `reuse` 层需要的那条顺序**。
- 另外观察到 40 层 `TPB=1` 那次的挂起点在 `attn L0/L1`（`148/148 SM entered`、`task0 started 0`），
  即卡在栅栏之后的派发上；而 6 层那次是算错不挂。同一配置两种表现 -> 仍是竞态。

**含义**：`e2e_decode.py` 这条服务形态的纯静态链现在既不能用于精度验收，也不能用于 TPOT 验收。
`full_decode.py`（单进程 host-reset，每层 host 等待）才是当前唯一"能算对"的路径，
所以精度验收在它上面做（§12.40/12.41 的 0.87x 带就是它给的）。

### 12.42.5 由此确定的优化顺序

1. **先修服务形态的链**（`TPB=1` 在 `attn L3` 的 NaN）：不修就没有可信的 TPOT 与精度闭环，
   而且 PDL 与栅栏的冲突（`tpb_flag` 的 assert）意味着现在服务端跑的 `-DSTATIC_NO_START_BARRIER + PDL` 组合
   实际上没有任何内核内的跨 rank 顺序保证。
2. **再压每次 launch 的固定开销**：80 次 launch，其中 ~92% 的成本与 32K 无关。
   可动的地方按收益排序：(a) 融合 sublayer，把 80 次降下来；(b) 缩短依赖跳数（`wo_a/wo_b`、`wq_a/wq_b` 的拆分）；
   (c) indexer 层那 8×~40 us（唯一真正的 32K 项）。
3. **精度验收标准不变**：每一步都必须仍在参考自身带宽内（§12.40.5 的 0.87x），argmax 343 不变。

### 12.42.6 复现命令（本篇的全部数字）

```bash
# 位置相关性（3 层 / 21 层，只改 --pos）
python full_decode.py --checkpoint $CK --gpus 8 --layers 21 --pos 1024  --tpot 3 --steps 0 --out /tmp/tpot_rx1024
python full_decode.py --checkpoint $CK --gpus 8 --layers 21 --pos 31906 --tpot 3 --steps 0 --out /tmp/tpot_rx31906
# 分期与 per-node（无需 profiler / 需 timing build）
python full_decode.py ... --probe --steps 0
python full_decode.py ... --probe --profile-build --steps 0
# 正向对照（§8.1 缺的那一格）
DSV41_STATIC_TPB=1 DSV41_E2E_LAYERS=6 torchrun --nproc_per_node=8 e2e_decode.py ...
```

## 12.43 服务形态 `attn L3` 的 NaN 结案：`sparse_attn` split 组合的 partial 违反单读者契约（2026-11-28）

§12.42.5 第 1 条（"先修服务形态的链"）已修。根因不是 KV、不是 attention 算术、也不是 `wo_a`，而是
**`REARM`（读者把 0xFF sentinel 写回去给下一次 launch）的单读者假设在 split 组合里不成立**。

### 12.43.1 先排掉的（都是可复现的否证，不是推断）

| 假设 | 实验 | 结果 |
|---|---|---|
| KV cache 被 0xFF 污染 | `DSV41_E2E_ZERO_KV=1`（`compress_cache`/`index_k`/`window_cache` 的值与 scale 全清零） | L3 仍 NaN → 否证 |
| `wo_a` GEMM 少算/漏算 | 指针普查：`o_q`(0x…6000) 与 `wo_a_part`(0x…8000) 相邻但**不重叠**，全层无地址别名 | 否证（此前的"别名覆盖"猜测不成立） |
| `o_q` 被重置成 0xFF | 生成代码声明表：`buf.31 out o_q 8192 reset -1 reader 0` → 既不被 kernel 重置也不被读者 re-arm | 否证 |
| in-kernel memset 本身 | **`REARM` 关掉、in-kernel reset 保留**（`-DSTATIC_NO_REARM`） | `all 4 attn + 4 moe x_out finite` → memset 无辜 |

关键观测：L3 launch 结束后 `o_q` 是 **8192/8192 字节全 0xFF**（bf16 全 `0xFFFF`，即 -NaN），
而 L2 只有 11 字节。`o_q` 没有任何重置路径，所以这是 `sparse_attn` 真的把 NaN 写满了 4096 个值。

### 12.43.2 违反契约的那一处

`sparse_attn` 的 split（`NS = 1 + NC/128 = 5`，`full_r2` 与 `reuse` 都是）里，每个 task `(t,g,s)`
轮询其余 `NS-1` 个 share 的 O partial，**然后各自把读过的那些 re-arm 回 0xFF**
（`sparse_attn.cuh:1430-1443`；q 的那份在 1445-1453 同理）。于是一个 `sparse_attn_part` 槽位有
**5 个读者、5 个写者**共用同一个 0xFF sentinel，`config.cuh` 的 `REARM` 契约（"each byte's ONE reader
re-arms it"）被违反：某个 task 的 re-arm 落在另一个 task 还没读完的时候，后者拿到的就是 0xFF → NaN。

这解释了为什么只有 reuse 层必现、full 层（L2）反而不炸：两者模板参数只有 producer 描述符不同
（`reuse` 无 `ckv_post`/`indexer_topk`），launch 更短、5 个 task 起跑更同步，竞态因此必现而不是偶发。
`full_decode.py`（host-reset，`REARM` 恒 false）一直干净，与此一致。

### 12.43.3 修法

`REARM` 的语义改为"**kernel_begin 在 start_barrier 之前重置**"——这正是 `full_decode` 已验证过的语义，
只是把 memset 从 host 搬进 kernel（服务形态仍然一次 launch 跑完，不需要 host 插手）：

- `python/mirage/mpk/static_megakernel.py`：`kind == "scratch"` 的 buffer（即 `sparse_attn_part`）声明成
  `by_reader=False`，进 kernel 的 reset 列表；
- `include/mirage/static_megakernel/config.cuh`：`REARM` 默认 false（`STATIC_REARM` 可opt-in回旧协议）；
- **交换缓冲（`y_sent` 等 `exchange=True`）保持 `by_reader`**：它们的写者是别的 GPU，本地 kernel_begin
  去填会把 peer 已经落地的贡献抹掉——实测直接挂死（不是 NaN）。

### 12.43.4 验收

- 4 层（`DSV41_E2E_LAYERS=4 DSV41_E2E_ONE=1`）：`all 4 attn + 4 moe x_out finite`
### 12.43.5 验收结果（全部实测）

| 检查 | 结果 |
|---|---|
| 4 层 finiteness | `all 4 attn + 4 moe x_out finite` |
| 40 层 finiteness | `all 40 attn + 40 moe x_out finite, logits (1, 129280) finite` |
| L3 的 buffer 普查 | `o_q` 8192/8192 → **7/8192** 字节 0xFF；`wo_a_part` 6144/8192 → **24/8192**（L2 是 19/8192，同量级）|
| 跨 launch 稳定性 | 4 层 `reps=1` vs `reps=3`：`x_out`/`streams_out` **逐位 100% 相同**（relRMS 0）|
| 与 host-reset 形态一致 | 4 层 serving vs `DSV41_STATIC_SERVING=0`：`x_out`/`streams_out` **逐位 100% 相同**（relRMS 0）|

最后一条是关键：修法没有改动任何算术，服务形态现在与 `full_decode` 那条已被 §12.40 验证过的路径**逐位一致**，
所以 §12.40.5 的 0.87x 带与 argmax 343 可以直接继承，不需要重新对 SGLang 建带。

复现：
```bash
# 服务形态 vs host-reset 的逐位对比（4 层）
DSV41_E2E_DUMP_LOGITS=out_s1.pt torchrun --nproc_per_node=8 e2e_decode.py ... --layers 4 --reps 1 --warmup 1
DSV41_STATIC_SERVING=0 DSV41_E2E_DUMP_LOGITS=out_h1.pt torchrun ... --layers 4 --reps 1 --warmup 1
```

### 12.43.6 修完之后的第一份可信耗时（serving，position 31906）

`--spans` 的 4 层：median step **1.09 ms**，8 次 launch 的 span 合计 238 us，**host gap 850 us（78%）**；
`attn3`（reuse）27.7 us 与 `attn1` 22.8 us 同级 —— 修 NaN 没有引入额外代价。

（注意：`--reps 1` 单样本会给出 29.9 ms 这种假数，`--reps 5` 起才有意义。）

40 层（position 31906，`--reps 5 --warmup 2 --spans`，这是修完之后第一份可信的 32K decode 数）：

| 项 | 值 |
|---|---|
| median step | **9186.8 us = 9.19 ms**（p10 7902, p90 24076）|
| 吞吐 | **109 tok/s** |
| 80 次 launch 的 kernel span 合计 | 3067 us（attention 1606 = 17%，MoE 1461 = 16%）|
| **launch 之间的 host gap** | **6120 us = 66.6%** |

p90 24 ms 的抖动来自同机共跑的其它训练进程（`nvidia-smi` 上 8 卡恒 100%），不是链本身。

**结论**：32K 下 kernel 真正在算的只有 3.07 ms，**6.12 ms（2/3）花在 80 次 launch 之间的 host 间隙**。
所以 TPOT 的主攻方向不是内核算术，而是 §12.42.5 第 2 条说的 launch 次数与 host 侧开销：
(a) 融合 sublayer 把 80 次降下来；(b) 缩短 host 侧的每-launch 固定成本（当前 ~76 us/次）。
`attn` 最重的 8 个（attn20/24/28/32/36 等 indexer 层，60-85 us）合计才 500 us，即使全砍掉也只有 5% 收益。

## 12.44 TPOT 第一刀：host 侧重复计算 window tables（2026-11-28，9.19 → 5.05 ms，1.82x）

§12.43.6 指出 66% 在 launch 之间的间隙。这一节把它量到底并砍掉。

### 12.44.1 归因（三次否证，最后落在 host 的 Python 上）

| 假设 | 实验 | 结果 |
|---|---|---|
| 间隙是 `dist.barrier()` | 读 `_sync()`：`TPB` 默认 1 | barrier 根本没执行 → 否 |
| 间隙是 kernel 内的跨 rank 等待 | span 的定义就是"最后一个 SM 过 start barrier"到"最后一个 SM 结束" | 跨 rank 等待已含在 span 内 → 否 |
| 就是 host 发不出去 | `DSV41_E2E_HOSTT=1` 逐层计时 | **host 6.2 ms / 8.2 ms step = 75%** → 是 |

再细到 per-launch：**attn 135 µs/次，MoE 只有 20 µs/次**，而 attn 的 kernel 只要 40 µs。
cProfile（`DSV41_E2E_PROFILE=1`）指名：`window_tables` **105 µs/次**，每层调一次。

### 12.44.2 根因

```python
def window_tables(w: dict):          # demo/deepseek_v4_1_csa2/demo.py:128
    p, t = w["positions"].long(), torch.arange(T, device=w["positions"].device)
    w["window_loc"].copy_(t * W + p % W)
    w["window_rows"].copy_(t[:, None] * W + (p[:, None] - torch.arange(W, device=p.device)) % W)
```

它是 **positions 的纯函数**，而一个 decode step 里 40 个 attention 层的 positions 完全相同 ——
同样的 132 个 int32 值被算了 40 遍，每遍 6 个 tiny CUDA op（每次 ~7 µs 的 host dispatch）。
这是纯 host 开销，不含任何算术：kernel 一侧毫无变化。

### 12.44.3 修法（用框架自己的机制，不改生成代码）

- `StepTables`（`dsv41_static_layer.py:397`）本来就是这个设计："one set for all the attention layers"，
  只是 e2e 这条路径没用。现在把 `positions` / `window_loc` / `window_rows` 一份共享，
  逐层用 `StaticAttn._rb()`（`bind_pools` 用的同一机制）重绑到共享 buffer；
- `run()` 里对 `tables_shared` 的层跳过那次重算（x 的拷贝保留：x 每层不同）；
- `step()` 每个 step 只调一次 `window_tables`。

开关 `DSV41_E2E_SHARE_TABLES`（默认开）。

### 12.44.4 验收

| 项 | 修前 | 修后 |
|---|---|---|
| 4 层 step | 1087.7 us | **891.8 us** |
| 40 层 step（pos 31906）| 9186.8 us | **5051.0 us** |
| 吞吐 | 109 tok/s | **198 tok/s** |
| host per step | 6.2 ms（75%）| **1.79 ms（35%）**|
| attn 的 host 侧 | 135 us/launch | **26 us/launch** |
| kernel span 合计 | 3067 us | 3073 us（不变，符合预期）|

正确性：
- 4 层 `x_out` / `streams_out` 与修前 **逐位 100% 相同**（relRMS 0，`/tmp/e2e_share/out_share.pt` vs `/tmp/e2e_fx3/out_s1.pt`）；
- 40 层 `all 40 attn + 40 moe x_out finite, logits (1, 129280) finite`，**argmax 343 = golden**（§12.40 的验收标准）。

### 12.44.5 顺带修掉的工程坑

`MIRAGE_STATIC_BUILD_CACHE` 之前没设，导致每次跑都把同一批层重编译 **640 次**（8 rank × 80 层，无共享），
单次运行要等 9 分钟。现在统一用 `/opt/dlami/nvme/saboxu/static_cache`：同一层只编一次，其余 rank 命中缓存。

### 12.44.6 下一刀（剩余 5.05 ms 的构成）

- kernel span 3073 µs（61%）：attention 1609 + MoE 1465
- host 1785 µs（35%）：attn 26 µs + MoE 18 µs，每次 launch 的 Python/CUDA 固定成本
- 结论：**host 已从瓶颈变成次要项，kernel 重新成为主导**。再往下要么走 CUDA graph 抹掉这 ~1.8 ms
  的 per-launch 成本，要么融合 sublayer 减少 launch 次数（80 次）。

## 12.45 e2e_decode 的 Engram 层是错的（已修）；CUDA graph 逐位正确但**不幂等**（2026-11-28）

这一节有两个结论，第二个是否定的。先说不利观感的那一半容易读错的地方：**§12.40 那条 logits 0.87x 是
`full_decode` 的链，不是 `e2e_decode` 的**。e2e 的链当时是 1.35x，带外。审这一节的时候不要把它们混起来。

### 12.45.1 缺陷：Engram 层只换了 x，没换 R

`engram_gate` 会把一层残差**整体替换**（§12.31）。所以 layer L 的 attention 读的是 R''（kernel 的
`streams_eg`），不是 R'（`streams_out`）。`full_decode.py` 两处都做了：

    Rs = [wg["streams_eg"] if "streams_eg" in wg else wg["streams_out"] for wg in wsm]   # :498
    ws[r]["x"][:1].copy_(eg_next_x(src[r]["streams_eg"], L, devs[r])[None])              # :731

`e2e_decode` 只做了第二件（host 重算 x），**第一件一直用 `streams_out`**。后果是确定的、不是噪声：

| 层 | e2e vs 参考 | chain_ok vs 参考 | 判定 |
|---|---|---|---|
| L0 | 2.647e-03 | 2.647e-03 | 逐位相同（L0 无 Engram） |
| L1（第一个 Engram 层的 attention） | **5.594e-01** | 9.186e-03 | e2e 差 60 倍 → 错的是 e2e |
| L1 的 moe | **5.413e-01** | 1.394e-02 | 同上 |

修法一行（`e2e_decode.py`，`moe[i].run` 之后）：

    R = (moe[i].w["streams_eg"] if "streams_eg" in moe[i].w else moe[i].w["streams_out"]).view(T, HC, HID)

修后**L0–L19 与 `chain_ok` 逐位相同（relRMS = 0）**。L20 起有 4e-3–8e-2 的差，那是两条链在混沌区的合法散布，
不是缺陷：L20–L39 上 e2e 反而比 chain_ok 更贴近参考（0.87–0.95x，见 §12.45.3）。

### 12.45.2 e2e 自己是可重复的（这条要单独验证，否则上面的对比没有意义）

同一配置两次独立启动（`/tmp/e2e_chain2`、`/tmp/e2e_chain3`）：**160 个 `x_out`/`streams_out` 全部逐位相同，
logits 逐位相同**。所以 §12.45.1 的差异是两条链的差，不是同一条链的抖。

### 12.45.3 层级别：修后与已验证的链**评分完全相同**

`cmp_band.py`（3 次参考自建带，`--slack 1.5`），只取两边都有的 77 行（apost + mpost）：

    chain        inside   最差有限比率
    chain_ok     64/77    3.62（L2 apost）
    e2e（修后）  64/77    3.62（L2 apost）

**逐行一致到最差那行都相同**，所以 e2e 现在与 `chain_ok` 在层级别等价；`e2e_chain` 的 64/77 与
`chain_ok` 的 101/117 不是同一分母（e2e 的 dump 没有 `_apost_x` 那份记录），不要拿来互相比较。

logits：修前 **1.35x（带外）**，修后 **1.12x**，argmax 343，top-3 = [343, 764, 14662] 与参考一致。
参考的带只有 2 个样本（4.493e-02），由参考自己的第 4/5 名在跳（§12.40.5）可知这条带偏窄，
所以 1.12x 与 0.87x 是同一个量级的抽样差，不是质量差。

### 12.45.4 CUDA graph：单次 replay 逐位正确，**多次 replay 不是**（否定结论）

`DSV41_E2E_GRAPH=1`：把一步拆成每段一个 graph（在 2 个 Engram 层断开，因为那里的 host 重算需要真同步），
实测 **4.49 ms / 223 tok/s**（40 层，pos 31906），对比 launch 链的 **5.01 ms / 200 tok/s**。

单次 replay 的等价性是**逐位验证过的**：160 个 `x_out`/`streams_out` + logits 与 launch 链完全相同
（`/tmp/e2e_graph` vs `/tmp/e2e_chain2`）。但**第 2/3 次 replay 起不再相同**：

    非 graph，13 步          13/13 逐位相同（logits 指纹全同）
    graph，13 步             第 3 步起漂移，logits 1.12x -> 2.76x
    graph，4 层             只有最后一层差，且第 2..7 步彼此完全相同

最后一行是关键判据：4 层下**第 2 步之后完全稳定**，说明不是一个累积误差，而是进了**另一个不动点**；
40 层下「首次漂移出现在第几步」还随 warmup 数变（warmup=3 时第 3 步、warmup=1 时第 2 步），所以这是
**竞态**，不是 §12.43 那种确定的奇偶错误。

嫌疑集中在 reset 协议里那些**每次 launch 变化**的设备端状态：`core.cuh kernel_begin` 的 `gen`
（读 `g.gen_ptr`、写 `dbuf_par`）与由它派生的 `rg_set()`（交换区第 n%2 套）。host 逐个 launch 时，
host 的 ~20 µs 间隔在给 rank 排序；graph 把整段一次提交，排序只剩 kernel 内那个 `>= gen` 的 TP barrier，
于是各 rank 可以稳定地停在另一个不动点上。

**所以 graph 不进主线**：默认关，代码里与这里都写成"实验性、不幂等，别引用它的数字"。
要救它，方向是让每次 replay 的 launch 序号归零（显式 `cudaMemsetAsync(gen_ptr, 0)` 每段一次），
这样每次 replay 与第一次逐位相同；但 `gen_ptr` 在模块内部（`g_st[0]`），Python 侧够不到，需要加一个导出。

### 12.45.5 剩下的 5.01 ms 是什么

    kernel span 合计   3076 µs（61%）：attention 1609 + MoE 1465
    核间空隙           1935 µs（39%）：80 次 launch，每次 ~24 µs

graph 只买下 516 µs，说明**空隙不是 host 提交慢**（host 已降到 ~1.0 ms，其中 0.8 ms 是 2 次 Engram 的真
`cuda.synchronize`，那两次 graph 也装不进去）。空隙是 GPU 侧的：每次 launch 的 prologue + 8 rank 在
TP barrier 上的会合。要再往下要么做 sublayer 融合（把 80 次 launch 变少），要么让 TP barrier 的会合更便宜；
**PDL 不能用来填这个空**——它与 TP start barrier 在设计上互斥（`dsv41_static_layer.tpb_flag` 里是硬断言：
barrier 的正确性前提就是 stream 串行）。

### 12.45.6 验收

| 项 | 值 |
|---|---|
| 40 层干净基线（instrumentation 全关） | **5.01 ms / 200 tok/s** |
| 层级别证书（vs chain_ok，同一 77 行） | **64/77 逐行相同，最差 3.62** |
| L0–L19 vs chain_ok | **逐位相同（relRMS 0）** |
| logits | 1.12x（2 样本带），argmax **343** |
| e2e 自身重复性 | 160/160 逐位相同，logits 逐位相同 |
| CUDA graph | 4.49 ms，单次逐位正确，**多次 replay 不幂等 → 不进主线** |

### 12.45.7 `step()` 的重构是逐位中性的（否则上面所有数字都不可信）

为了做 graph，`step()` 被拆成 `_sublayer(i)`（纯发射）+ `_segments()` + `capture_graph()`。这条重构必须证明
它没有动算术，判据是：**重构后的非 graph 链，与重构前那条 40 层链逐位相同**（`/tmp/e2e_refac` vs
`/tmp/e2e_chain2`）：

    160 个 x_out / streams_out            0 个不同
    logits                                逐位相同
    argmax                                343
    层级别证书（cmp_band，同一 77 行）      64/77，最差 3.62（与重构前逐行相同）
    40 层 step                            5.01 ms -> 4.99 ms（200 tok/s，在抖动内）

`DSV41_E2E_SIG`（默认关）保留在树里：它每一步对每层的输出做一次 bit 指纹，是"这一步和上一步是不是同一个
计算"的唯一便宜判据——§12.45.4 靠它定位到 `attn3.x_out`。

## 12.46 host 已经不是关键路径了：5 个实验，1 个正收益（2026-11-28）

§12.45 之后剩下 39% 是"核间空隙"。这一节把"空隙是不是 host 提交慢"这个问题**永久关掉**：改了 3 处 host 路径，
每处都逐位验证，host 时间降了 21%，而 step 时间**没动**。

### 12.46.1 实验表

| 实验 | 改了什么 | host per step | step (中位数) | 逐位 |
|---|---|---|---|---|
| 基线（§12.45.6） | — | 1790 µs | 5.01 ms | — |
| **x 重绑** | attn 不再 `copy_` 上一层的 x_out，改为 `_bind_x` 原地读 | **1452 µs** | **4.94 ms** | 0/160，logits 逐位 |
| + stream 缓存 | `current_stream()` 每步缓存一次，不再每次 launch 建 Stream 对象 | 1479 µs | 4.98 ms | 0/160 |
| + 预算视图 | 每层的 R/x_out 张量建层时解析一次（160 次 dict + 160 个 view/步 → 0） | **1415 µs** | 4.99 ms | 0/160 |
| L2 权重预取 32 MB | `STATIC_L2_PREFETCH`（glm53 的层一直开着，DSV41 从没开过） | — | **5.02 ms（更慢）** | 0/160 |
| poll 暂停 200 ns | `DSV41_POLL_NS=200`，降低 148 个 SM 自旋时的争用 | — | 4.97 ms（噪声内） | — |

**读法**：host 从 1790 µs 降到 1415 µs（−21%），step 从 5.01 到 4.94/4.98/4.99（噪声带 ±0.05 ms）。
三处 host 优化全部逐位中性，但**都没有换来时间**，这就是结论：host 已经不和 GPU 抢关键路径了。

### 12.46.2 为什么 host 降了 21% 而 step 不动

因为 GPU 侧的核间空隙不是 host 造成的。判据是 CUDA graph（§12.45.4）：它把一步的 80 次提交压成 3 次，
host 几乎归零，而 step 只从 5.01 掉到 4.49 —— 说明 39% 的空隙里只有 **~0.5 ms 是提交**，其余 **~1.45 ms 是
GPU 自己的**（每次 launch 的 grid 分发 + 148 个 CTA 的 resets + 8 个 rank 在 TP start barrier 上的会合，
平均 18 µs/launch）。

这也解释了 §12.44 的判断为什么需要修正：那里说"host 1785 µs（35%）"暗示 host 是个大头，实际上**它已经不在
关键路径上了**，只是还在花时间。

### 12.46.3 两个被排除的方向

- **PDL**：与 TP start barrier 设计上互斥（`dsv41_static_layer.tpb_flag` 的硬断言）。PDL 让下一个核的 prologue
  与上一个核重叠，而 prologue 里的 resets 写的是上一个核的消费者可能还在读的缓冲区 —— 那正是 §12.43 的 L3 NaN。
- **自旋争用**：`DSV41_POLL_NS=200` 无变化，所以 barrier 的代价不是 148 个 SM 自旋打满 L2，而是会合本身的延迟。

### 12.46.4 L2 预取的负结果（-3%）值得单独记

`l2_prefetch_weights` 让 prologue 里已经在等的 CTA 顺手把本层头几个 GEMM 的权重拉进 L2。纸面上正好填 §12.46.2
那 18 µs，实测**更慢**（5.02 vs 4.94）。原因在它的 docstring 里已经写了：没有 PDL 时 CTA 要等上一个核结束才
启动，可用的窗口只有 arrive 等待那一小段，剩下的预取流量反而抢了上一个核尾部的带宽。

顺带修了它的一个覆盖洞：`l2_prefetch_weights` 的节点表原来只认 `gemm_tile` / `scaled_gemm_tile`，DSV41 的注意力
层用的是 `gemm_fp8_tile`，于是注意力层一个权重都取不到（只有 MoE 的 gate/共享专家生效）。已加上，逐位中性。

### 12.46.5 结论：下一步只能是减少 launch 次数

现在 4.99 ms 的账：

    kernel span        3076 µs（62%）：attention 1609 + MoE 1465
    核间空隙          ~1900 µs（38%）：其中 ~500 是提交（可被 graph 买下）、~1450 是 GPU 侧
    host               1415 µs：不在关键路径上（§12.46.1 的三次实验）

两条都被"80 次 launch"卡着：空隙按次计费，而单次 kernel 只有 38.5 µs（148 个 SM 上是一条 10 个节点的
依赖链，T=1 时每个 GEMM 都是 [1, 5120]×[5120, N] 的瘦乘，延迟主导）。

所以真正的下一刀是 **sublayer 融合**：把 attn(L) 与 moe(L) 编进同一个 kernel（一次 launch、一次 grid 分发、
一次 resets、一次 TP barrier），launch 数 80 → 40，纸面 ~720 µs；同时 MoE 的 gate GEMM 可以在 attn 的尾部
节点还没结束时就开始。它需要把两个 `build_graph` 挂到同一个 `StaticMegakernel` 上（张量名冲突、buffer slot
合并、两个 sum_store 的交换槽位、plan 的合并死锁检查），是个结构性改动，不是这一轮能验完的。

## 12.47 64/77 的解剖：它不是"哪里算错了"，是路由器在这个 token 上本来就站在悬崖边（2026-11-28）

`cmp_band.py` 的 64/77 一直被当成"还有提升空间"。这一节把它逐行拆开，结论是：**除了一处，其余 12 行都是
结构性上限**；而那一处的原因也不是 kernel 写错了。

### 12.47.1 13 行的归类

| 类 | 行 | 数 | 性质 |
|---|---|---|---|
| A | L0 apost, L1 apost, L1 mpost | 3 | band **恰好 = 0**（参考自己逐 bit 一致），任何非零差异都是 inf |
| B | L2 apost, L2 mpost, L3 apost | 3 | 早层 band 极小（L2 仅 5.7e-3），我们 2.06e-2 |
| C | L7 mpost, L8/L9/L10 的 apost+mpost | 7 | 单一注入后衰减，band 长上来后比值回落 |

C 类是全 40 层扫描里唯一的大跳变：

    layer  apost: ours  band  ratio  jump  |  mpost: ours  band  ratio  jump
        6  3.097e-02 2.704e-02  1.15 1.05  |  3.390e-02 3.200e-02  1.06 1.09
        7  3.671e-02 3.234e-02  1.14 1.19  |  8.685e-02 3.526e-02  2.46 2.56 <<<
        8  8.325e-02 3.584e-02  2.32 2.27  |  8.220e-02 3.724e-02  2.21 0.95
        ...
       13  6.772e-02 5.698e-02  1.19 0.96  |       -           -     -    -
       14  6.398e-02 5.507e-02  1.16 0.94  |  6.480e-02 9.845e-02  0.66 0.92
       20  1.158e-01 1.297e-01  0.89 1.01  |  1.097e-01 1.245e-01  0.88 0.94
       39  8.906e-02 1.048e-01  0.85 0.95  |       -           -     -    -

三个要点：
1. 注入是**一次**的：`moe[7]` 一步 1.06 → 2.56，之后误差**保持并衰减**（8.2 → 7.8 → 7.7 → 7.4 → 6.5e-2），
   而 band 稳步增长；L13 之后 24 行全部回到 0.84–0.92，即我们比参考们彼此之间还要靠近。
2. 这**不是** §12.38 那种 kernel 缺陷：真正的缺陷（MoE buffer）是 0.6–1.6 并且**持续增长**，而这里在衰减。
3. 到三个参考的距离几乎相等（8.695 / 8.667 / 8.685 e-2）——混沌噪声会让三个距离散开，等距说明这是
   三个参考聚簇之外的一个**系统性**偏移点。

### 12.47.2 那一处是路由：L7 的 6/7 名分差是全 40 层最紧的

用我们自己的 FFN 输入（`attn7_x_out`）+ 自己的 gate 权重算一次 top-6：

    L6: margin6-7 6.315e-03 (margin/score 5.45e-04)   输入 3e-2 噪声下 24/24 稳定
    L7: margin6-7 5.503e-04 (margin/score 4.47e-05)   输入 3.7e-2 噪声下 11/24 稳定   <-- 掷硬币
    L8: margin6-7 5.326e-02 (margin/score 4.30e-03)   输入 3e-2 噪声下 24/24 稳定

全 40 层排序，L7 的 margin/score 是**第 1 名（最紧）**：

    1  L 7  4.468e-05      4  L18  1.352e-04      7  L 0  2.072e-04
    2  L36  5.946e-05      5  L27  1.464e-04      8  L24  2.722e-04
    3  L30  1.163e-04      6  L32  1.810e-04      median 9.88e-04

而 L7 的输入偏差已经是 3.7e-2（`attn7` 的 apost 距离），比它自己的判决余量大 **828 倍**。
另一个观察：只有 L7 在 **0.5%** 输入噪声下就开始翻转（12%），其它层在这个量级几乎都是 0。

### 12.47.3 但"路由脆弱"是普遍的，不能拿它当"只有 L7 有问题"

噪声扫描（每层 16 次、固定种子，比较 top-6 集合）：

    层   margin/score | 0.005 0.010 0.020 0.030 0.035 0.040 0.050 0.070 0.100
    L30  1.163e-04 |  0.00  0.06  0.25  0.50  0.38  0.56  0.56  0.62  0.69
    L 7  4.468e-05 |  0.12  0.38  0.38  0.31  0.44  0.31  0.81  0.50  0.62
    L 0  2.072e-04 |  0.00  0.00  0.00  0.06  0.00  0.06  0.19  0.19  0.44   (never)
    L 1  1.082e-03 |  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.06  0.06   (never)

也就是说：在 3% 输入噪声（≈我们实际的漂移量级）下，**大多数层**的 top-6 都有 30–50% 的翻转率。我们只观察到
一处大跳，是因为实际的偏差向量是一个**特定方向**，未必落在 gate 判决最敏感的方向上；而随机各向同性噪声会
高估敏感度。所以这张表给出的正确结论是"路由器普遍站在边界上"，不是"只有 L7 坏"。

（这道扫描的第一版有 bug：`t8` 取的是分数张量而不是 id 列表，于是所有层都报 100% 翻转。已修正后重跑，
上表是修正后的数据。教训：这种集合比较必须直接打印一次 ids 做 sanity check。）

### 12.47.4 A/B 两类为什么不可修

- **A 类（band = 0）**：实测 L0 apost 与参考逐 bit 比较，`20480` 个 bf16 里 **13.89% 不同**，p99 ≈ **9 ulp**：

      L0 apost: relRMS 2.647e-03  diff 2845/20480 (13.89%)  ulp p99 9  bit-exact 17635/20480
      L1 apost: relRMS 9.186e-03  diff 12674/20480 (61.88%) ulp p99 70
      L2 apost: relRMS 2.059e-02  diff 14554/20480 (71.06%)

  这不是"差一点点、抠一下就对"：14% 的元素跨了 ~9 个可表示值，是 5120 深归约的整段累加序差异。
  要在 band=0 的层上进带，必须与 flash_mla **逐 bit** 相等，也就是完整复刻它的 MMA tile 序 + split 归约序
  + P 的舍入点。§12.41 已经用 fp64 真值证明我们是同一精度档里的等距样本（0.999x），所以这不是"精度不够"，
  纯粹是"序不同"。
- **B 类（早层 band 极小）**：我们的**绝对**误差在 L2 是 2.06e-2，同一个量级在 L13+ 是 0.85x（band ~1e-1）。
  所以 L2/L3 判"带外"不是因为我们差，而是因为**参考自己的带随深度非平稳**（从 0 长到 1e-1）。任何不是
  L0 逐 bit 一致的实现，都必然在 L2–L3 带外。这是记分卡的伪影，不是实现的缺陷。

### 12.47.5 结论

- **C 类 7 行**：要修必须先把 L0–L6 做到逐 bit，否则 L7 的路由判决无法复现（余量 4.5e-5 vs 漂移 3.7e-2）。
  这不是"改个 kernel 参数"，是复刻 flash_mla 的归约序。
- **A 类 3 行 + B 类 3 行**：结构性不可达。
- 因此 **64/77 是本实现（非逐 bit 复刻）的上限**，而不是待修的 bug 清单。要突破它，唯一的路径是把
  `sparse_attn` 的累加序对齐到 flash_mla 的逐 bit 等价——收益是把这 6 行变成"带内"，代价是一个独立的、
  不小且不保证成功的方向。

这也把 §12.40.6 的"三层都达到了"讲得更清楚了：逐层 64/77 ≠ 我们比参考差，而是这个判据在早层把
"序不同"和"算错了"混在一起。真正判别实现的是 L13–L39 那 24 行（0.84–0.92），以及 logits 层
（0.87–1.12x，argmax 343）。

## 12.48 两条被否掉的路：graph（更快但错）和"sparse_attn 单 SM 是瓶颈"（不是）（2026-11-28）

这一节记录两个都**看起来该有收益、实测都不成立**的方向。两次否证都有确定的数字，写下来是为了不再重复投。

### 12.48.1 CUDA graph：4.50 ms，但数值在带外 4.3 倍

`DSV41_E2E_GRAPH=1`（3 段图，切在 Engram 层 1/14）在 40 层、32K 上实测：

| | launch 链 | CUDA graph |
|---|---|---|
| TPOT | 4.95 ms / 202 tok/s | **4.50 ms / 222 tok/s**（-9%） |
| `cmp_band.py` | 64/77，worst 3.62 | **12/77，worst inf** |
| logits | 0.87–1.12x 带内 | **4.22–4.33x 带外** |
| argmax | 343 | 343（top-5 集合全变） |

**9% 是拿错的数换的**。`DSV41_E2E_SIG=1` 给出漂移的形状：第 2 步就开始（attn3），第 3–8 步在
attn1/attn2/attn3 之间移动，**第 9 步起固定在同样 148 个张量上不再恶化**，同时 TPOT 稳定在 4.50 ms。
所以它不是发散，是**收敛到另一个不动点**——而那个不动点在参考带外 4.3 倍。

顺带一个对归因有用的事实：graph 把 gap 从 1881 µs 降到 1424 µs，**只降了 457 µs**，而 kernel span
一点没动（3066）。也就是说那 ~17.8 µs/launch 里只有 ~5.7 µs 是 launch 提交，剩下 ~12 µs 是 **GPU 侧的
CTA 分发/teardown**，graph 消不掉。要消它必须减少 launch 数（融合），不是换提交方式。

### 12.48.2 8 个慢层 = `index_source_layers`，但不是 bug

逐 launch 的 attn span 有两个独立的复现（`/tmp/e2e_40f`、`/tmp/e2e_bar2`，中位数分别 33.5 / 33.4）：

    attn 2   60.1 us  1.80x        attn20   83.9 us  2.51x
    attn 8   59.3 us  1.77x        attn24   69.8 us  2.09x
    attn14   59.6 us  1.78x        attn28   68.9 us  2.06x
    attn 0..39 其余        33 u

合计比常速层多 **275 µs（TPOT 的 5.5%）**。原因查清了：`index_source_layers = [2, 8, 14, 20, 24, 28, 32, 36]`
**逐项等于**这 8 层（`kv_source_layers = [2, 8, 14, 20]` 是其中前 4 个）。这 8 层真正计算 indexer
（对全部 compressed 位置打分 + top-k），其余 32 层 `reuse` 上一层的结果。成本随 `cap` 走：L20（ratio 1，
cap 32000）83.9 µs；L2/8/14（ratio 2，cap 16128）~60 µs。**这是设计，不是缺陷。**

### 12.48.3 否证：`sparse_attn` 只占 1 个 SM，但它不是瓶颈

plan 里 `sparse_attn` 在复用层是 `[1, 1, 1]`——**整个 MLA 只在一个 SM 上**（148 个里 147 个空着），
看着像一个明显的优化点。但它经不起一次对照：

    attn0  (swa, 只有 128 行 window，无 top-k 选择)   30.8 us
    attn1  (同上)                                     28.8 us
    attn3  (128 window + 512 选中 = 640 行)           33.2 us
    median (40 层)                                    33.4 us

KV 行数从 128 涨到 640（**5 倍**），时间只多 **~3 µs**。所以那 28–33 µs 不是行数/带宽受限，而是一条
**近乎固定的依赖链延迟**：`q_norm → kv_post → indexer_q → sparse_attn → wo_a → wo_b → sum_store` 在
同一个 SM 上串行，每步都要等 TMA / 上一个节点的数据。把 1 个 task 拆成 5 个（grid 的第 3 维本来是
`1 + NC/128` 的 split 轴，全量层已经在用 5 split）帮不上忙——供给不足的不是并发度，是链长。

同理，`sparse_attn` 的 task 粒度是「每 16 个头一个 task」，TP8 下每个 rank 只有 8 个头，按头再拆也不可行。

### 12.48.4 于是 TPOT 的账是这样

    4050 µs 的 kernel 工作
      1462  MoE：权重带宽地板（189 MB/层 ÷ 6 TB/s ≈ 31 µs，实测 35 µs/层）——没有空间
      1604  attn：其中 275 是 8 个 indexer 层的真实计算，其余是 ~33 µs/层的依赖链延迟
       504  TP start barrier（6.3 µs/launch × 80）
      1378  launch 提交 + CTA 分发（17.8 µs/launch；graph 只能拿回 457）

能动的只有最后一项，而它要求**减少 launch 数**——即 §12.46 末尾说的 sublayer 融合（80 → 40）。上面这条
否证也说明：融合的收益不会来自"让 pipeline 并发"，而是单纯地少 40 次 CTA 分发 + 少 40 次 TP barrier，
纸面 ~700 µs。

## 12.49 sublayer 融合逐项拆障：结构上建得起来，plan 必须用两个真实 hand plan 拼（2026-11-28）

§12.48 把 4.95 ms 归成"kernel 3076 + barrier 504 + launch 1378"，并指出只有减少 launch 数能动。这一节把融合
这件事从"纸面"推到**可执行的构造**：它不是设计问题，是四个具体缺口的工程问题，现在四个都解决了，剩下一项
（真实 harness 上的融合 kernel）已缩小到"把两个真实 hand plan 拼起来"。

### 12.49.1 四个缺口和它们的解法

| 缺口 | 现象 | 解法 | 状态 |
|---|---|---|---|
| buffer 槽位 | 两个子层的 `_new_buf` 共用一个计数，`config.cuh` 的 `MAX_BUFS=24` 不够 | `#ifndef STATIC_MAX_BUFS` 早已存在（`StaticMegakernel.max_bufs`） | 已有；融合两层实测 **38** |
| TMA map 槽位 | `MAX_MAPS=32` 是**硬编码常量**，两个子层的权重 map 放不下（实测 40） | 加 `STATIC_MAX_MAPS` 覆盖 + `StaticMegakernel.max_maps`，同 `max_bufs` 的形状：`config.cuh`、`_map`/`_gemm_map` 的两处断言、生成代码的 `#define`、`config_checks` 的 `static_assert` | **本轮新增** |
| 张量名冲突 | 两个 `build_graph` 都声明 `x`/`y`/`x_q`/`logit_part` | 前缀作用域（`s0.`/`s1.`，`fused_spike.scoped`）；`hand_plan` 按名查表，拼 plan 期间把 `compiler.tname` 的前缀剥掉 | 本轮验证：两个 hand plan 都能在带前缀的图上跑通 |
| 依赖边 | moe 的 `x` 必须等 attn 的 post；`attach_input` 造的是图输入、没有边 | `build_graph(..., bind=)`：`bind(k, w[k])` 返回 DTensor 就把该张量接成图内的边；attn 侧本来就把 `x_out -> streams_out` 注册进 `compiler.PRODUCER_ALIAS`，所以 moe 的 `x` 绑到 attn 的 `x_out` 就是一条真边 | **本轮新增**（两个 demo 各一个 hook，默认关闭） |

另外加了 `compiler.compile_plan(..., validate_grids=False)`：`candidate_grids` 是按**单个**子层的 map 推导的，
合并图上会把某些节点判成非法（实测 moe 的一个 `gemm_fp8_tile`：`(40,1,1)` vs 候选只剩 `(40,2,1)`）。融合的
plan 由真实 hand plan 给出，本来就不该再过这道推导。

### 12.49.2 实测：一个 mpk 上两张图建得起来

`fused_spike.py`（attention 层 3 + 它自己的 MoE，TP8，pos 31906）：

    s0 (attn): 25 buffer slots     s0 的 hand plan: 225 tasks
    s1 (moe) : 38 buffer slots     s1 的 hand plan: 280 tasks
    18 nodes, 505 tasks, 3.4 per SM（每 SM 的任务表放 62）    max_bufs 38  max_maps 40

即：**一次 launch 里放两个子层的 18 个节点、505 个 task，只占每 SM 3.4 个槽位**。任务表不是瓶颈（这是融合
最容易被担心的一项，实测差了 18 倍）。

### 12.49.3 唯一剩下的约束：plan 不能合成，必须拼两个真实 hand plan

第一版用"图顺序 + round robin"合成 plan，`nvcc` 直接拒：

    act_quant: "whole 128-column blocks, at most one pass of 2048 columns, K % NC == 0"
    gemm_fp8_tile: "fused act_quant: two jobs' stages in the ring at once"

原因是**每个 task 的 grid 和它自己的 ring/列切分是绑死的**（`act_quant` 的 `NC = K / grid.y`；`x_q` 在真实
构建里是 `grid [1,2,1]` —— 从 `schedules/schedule_gpu0.json` 读出来的，K=5120 时它就是这么编过的）。合成
plan 换个 grid 就换掉列宽，task 自己的 `static_assert` 立刻不成立。改成"两个真实 hand plan 按 SM 拼接"之后
（225+280=505），`act_quant` 的 K 分段与 ring 两条断言都消失了。

### 12.49.4 一个必须记下来的坑：`fused_spike` 的形状不是模型的形状

拼接后还剩 `act_quant_parts_task<K=5120, P=1, NC=5120>` 一条断言。定位到它是 **`oa_q` 那个 fused tap**：
spike 里 `oa_q` 是 `(5120,)`，而真实 TP8 构建里它是 `G*OL = 1024`（`act_quant.cuh` 的注释就写着"attention's
wo_a output is G OL = 1024 there"）。也就是说这条失败是 spike 自己喂权重的方式带来的（它走的是 plain demo 的
`layer_tensors` + `state_tensors`，没有 mHC 的 `streams`，也没按 variant 取 oa_q 的分片），**不是融合机制的
限制**。真实 harness 的 `attn3` 用同一份代码、同一个 grid `[1,2,1]` 是编过的（上表）。

结论：**融合原型不能建在 plain demo 上，必须建在 `dsv41_static_layer.StaticAttn/StaticMoE` 的权重与 variant 上**
——那条链的 `build_graph` + `hand_plan` 已经自洽，融合只需在它之上换成一个 mpk、两次 build_graph、拼两个 plan。

### 12.49.5 顺带的两份实测（今天这台机器）

    40 层，pos 32344，--reps 5 --warmup 2，无 --spans：median **4.3228 ms / 231 tok/s**（p10 4300.6）
    + DSV41_STATIC_IS_TCGEN05=1：median **4.3055 ms / 232 tok/s**（p10 4292.3，**p90 4468.6 vs 8569.7**）

`IS_TCGEN05` 的 median 在噪声内（-0.4%），但把 p90 的 8.5 ms 尾巴收掉了（那一档是首步后的抖动）。它改的是
indexer 的打分路径，`act_quant`/`sparse_attn` 的数值不变（`IS_TCGEN05` 注释："the same y bits"），要作为默认
开关还需要跑一遍 §12.40 的证书。基线比记录里的 4.95 ms 低，是因为那份数字带 `--spans` 且当时机器在编译。

**空隙的归属再确认一次**：serve 形态下 `_launch` 只是 `sk.launch_stream(stream, pdl)`（`static_harness.py`），
没有 host 侧对齐；80 次 launch 的 1.24 ms 空隙（4.32 - 3.08）因此全在 GPU 侧：CTA teardown + 下一次的 grid
分发 + 6.3 µs 的入口 barrier。所以"减少 launch 数"是唯一的路，融合的纸面收益仍是 ~40 × 17 µs ≈ 680 µs。

### 12.50 融合跑通了：一层 attn+MoE 一个 kernel，**bit-identical**，每层省 14.5 µs

`fused_layer.py`（新文件，torchrun 8 进程、真实权重、mHC 开）在 L8 `full_r2` 上把 attention 和它的 MoE 合成
一个静态 megakernel 并跑通。12.49 列的四个阻塞点是前提；这一节记的是把它真正跑起来之后剩下的**三个**问题，
以及实测。

**结构账**（实测打印）：45 slots / 28 maps = attention 一个子层；融合后 **65 slots / 56 maps / 34 nodes /
716 tasks / 4.8 tasks 每 SM**（表容量 62）。两个 plan 就是两个子层**自己已经验证过的 schedule**
（`attn8_t1/schedules/schedule_gpu0.json` 与 `moe8_t1/...`），按 SM 拼接、graph_idx 按**顺序**重映射
（子层的 idx 不是 0..n-1，taps 和 scratch 夹在中间）。

**第一个问题：MoE 的 x 不是图输入。** `act_quant_layer` 只接受"图输入"或"slots GEMM 的 K parts"；mHC 打开时
MoE 的 x 是 attention 的 `hc_post_combine` 输出（bf16 [T,K] 的普通 store）。它的 p-is-None 分支本来就是**按指针
读**，所以修法不是新 kernel 能力，而是把条件从"p 为 None"放宽成"不是 slots-GEMM"：依赖边由输入 map 的切片
（grid 切 token）自动导出，`dry`/`cost_inputs` 只在 slots 路径上。这一条让 MoE 的头节点真的等 attention 的 post。

**第二个问题：缓冲要按 kernel 自己的声明分配。** 融合图里同一个张量名会挂两个 buffer slot（GEMM 的 K parts
槽 + 它的 [T,N] 视图），按 tensors 名建 `{name: tensor}` 字典就会把 20480 B 的槽指到 5120 B 的张量上，init 的
`cudaMemset` 直接 `cudaErrorInvalidValue`。现在从生成的 `layer.cu` 里读 `"buf.N", "<kind> <name> <bytes>
<reset> <by_reader>"` 来分配（这也顺带证明了：融合后每个 `a.*`/`m.*` 的 out/tensor 槽，**尺寸逐字节等于**两个
子层各自构建时的尺寸）。

**第三个问题（最隐蔽）：计数器行数。** 两个子层按"地板"建（96 slots / 96 maps / 48 counter lines），但
`dsv41_mhc_nodes` 返回时会把 `max_bufs` 降到当前用量，所以我在地板撤掉后只重设了 `max_bufs`/`max_maps`，
`STATIC_NODE_COUNTER_LINES` 落回 config.cuh 的 **16**——而图里有 **34** 个节点带计数器，第 17 个开始写越
`G::cnt`（16 个字），表现为**非法内存访问**（不是 static assert，因为 `config_checks` 只比较生成值和
`max_counter_lines` 自身）。修好后 `max_counter_lines = mpk._counter_lines`。

**实测（L8，pos 31906，TP8，20 reps）**：

    两次 launch（attn 108.5 µs + moe）: median 108.5-109.2 µs（min 106.9）
    一次融合 launch                    : median  93.8- 94.7 µs（min  91.3）
    -> 每层 -14.5 µs（-13%），40 层 = **-0.58 ms**，4.32 ms 的步 -> ~3.74 ms

**正确性（这条链是精确的，不是"在带内"）**：融合 kernel 跑完后，`a.x_out` **5120/5120**、`m.x_out`
**5120/5120**、`m.streams_out` **20480/20480** 与"attn kernel 然后 moe kernel"那条两 kernel 链**逐 bit 相同**。
连续 **5 次** launch 每一次都相同 → in-kernel reset 在融合后仍然幂等（这正是 §12.44 的 CUDA graph 做不到的，
也是它被否掉的原因）。

**对照实验**：把 MoE 的图去掉、只留 attention 走同一个 builder（`FUSE_NO_MOE=1`），结果同样 bit-identical 于
两 kernel 链——说明这条 builder（一个 mpk、scoped 前缀、schedule 拼接、声明式缓冲）本身没有引入偏差，1 个
子层和 2 个子层的差别只在这三个问题上。

**还剩的**：40 层全都融合（每层一个融合 build），然后拿 §12.40 的证书（64/77 band + logits 1.12x + argmax 343）
在整步上复核。每层 -14.5 µs 是从 L8 这一层量的；L0-L7 的 attention 是 swa 变体、内核更小，空隙占比更高，
收益只会更大（纸面 ~680 µs），但要用整步实测确认，不能用单层外推当结论。
