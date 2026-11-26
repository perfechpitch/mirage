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

### 4.3 每次 launch 被 reset 的缓冲区，填充值就是 NaN

attn3 的生成代码（`build/sgl_static_dsv41/rank0/attn3/build/layer.cu`）：

| 缓冲区 | bytes | reset 字节 | by_reader |
|---|---|---|---|
| `sparse_attn_ml` | 10240 | **255** | 0（内核每次 launch reset） |
| `wo_a_part` | 65536 | **255** | 0（内核每次 launch reset） |
| `sparse_attn_part` / `wq_a_part` / `wq_b_part` / `wkv_part` / `wo_b_part` | — | 255 | 1（读方自己重新武装） |

`0xFF` 当 float 就是 NaN，这是本代码库的"尚未写入"哨兵（`runtime.cuh` 的 `valid16` / `validf4`
都在判它）。所以读到它就是读到 NaN，与 `l2_nan.log` 里 `x finite=True / y NaN` 完全吻合。

### 4.4 为什么首个 NaN 落在第一个 reuse 层

reuse 层是**第一个消费另一层产物的层**（读 L2 的 `compress_cache` + `compress_idx`）。
只写自己 buffer 的层（L0–L2）即使漂移也看不出来；一旦跨层传值，漂移立刻变现。
首次 NaN 在 L3/L4 之间移动，取决于启动时序 —— 竞态特征，而非确定性算错。

### 4.5 修法（已落到代码）

内核起点栅栏 `-DSTATIC_TP_BARRIER`，位置在 `core.cuh:611–634`（584 那条分支内），
**在 reset 循环（644）之前**。每个 SM 的 thread 0 等所有 rank 都发布了本次 launch 的编号
（`EXCHANGE_HELLO + rank*16`，和 host 侧 `static_align_kernel` **同一个槽位、同一语义**），
然后 `__syncthreads()` 放行本 SM 的其余线程。

> **位置必须在 reset 之前，这是我自己第一版写错的地方。**
> 放在 reset 之后不只是"少保护一点"：reset 会把本卡自己的 `wo_a_part` / `sparse_attn_ml`
> memset 成 `0xFF`，而这些**正是 peer 要读的**。晚到的 peer 不但拿到 NaN，它轮询的那份数据被抹掉后
> **永远等不到** → 死锁，比 NaN 更糟。
> 放在 reset 之前语义才成立：一个 rank 进入 launch n 就证明它已跑完 launch n−1（同 stream 串行），
> 所以"所有 rank 都进入了 n"等价于"没有 rank 还在消费 n−1 的产物"，此时 reset 安全。
> 顺带把漂移钳到 ≤ 1 次 launch —— 恰好是两套 exchange 需要的不变量。

`multimem.st` 会写包括本卡在内的所有副本，因此不会在 `r == g.rank` 上自旋死锁。

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

### 8.1 证实"缺栅栏"就是根因（最便宜的一刀）

> 判读入口：banner 打印 `TPBAR=` 与 `TPB=`。看到 `check=False` 且两者都为 0 的日志，
> 就**不能**用它论证"kernel 算错了"（那种配置本来就缺跨 rank 顺序）。

最小复现是 **`MHC=0` + 纯静态链**（`l2_nan.log` 的配置，NaN 在 L4）。
现在 `TPB` 默认开，所以这条命令本身就是验证；再用 `DSV41_STATIC_TPB=0` 做反向对照，
NaN 应当**重新出现** —— 一正一反才算证明：

```bash
cd /opt/dlami/nvme/saboxu/mirage/demo/deepseek_v4_1_sglang
export PATH=/opt/dlami/nvme/saboxu/venvs/sglang/bin:$PATH PYTHONPATH=$PWD/boot
export DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_POOL=1
export DSV41_STATIC_AR=0 DSV41_STATIC_MHC=0 DSV41_STATIC_TPBAR=0
export DSV41_STATIC_MAX_POS=33000 SGL_TP=8
for tpb in 1 0; do
  DSV41_STATIC_TPB=$tpb python l2_check.py \
    --model /opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017 \
    --row 0 --tokens 1 --batch 8 > /tmp/tpb$tpb.log 2>&1
  echo "=== TPB=$tpb"; grep -aE "attn L[0-9] step .*finite|MoE L[0-9] step .*finite" /tmp/tpb$tpb.log | head -12
done
# 判据: TPB=1 全 finite=True 且 TPB=0 重现 L4 NaN -> 根因确认
```

注意：改了 build flag 会**作废已有 80 个 .so**（`-DSTATIC_TP_BARRIER` 进 nvcc 命令行），
第一次跑要重编译约 1.4 min × 80。两轮之间 `DSV41_STATIC_TPB` 切换同样会各编译一次，
所以 8.1 一共是两套 build（约 2 小时）。想省时间就先只跑 `TPB=1` 那轮。
```

若证实，修法是**把 `TPBAR` 的默认值改成常开**（不再依赖 `MHC`）：

```python
TPBAR = os.environ.get("DSV41_STATIC_TPBAR", "1") != "0"
```

（代价：每个 sublayer 一次 NCCL barrier；单步 80 个 sublayer 就是 80 次，bs=1 延迟敏感，所以
正确性先全开，性能那轮再用 `DSV41_STATIC_TPBAR=0` 做对照并记录差异。）

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

`full_decode.py` 是单进程旁路，它产出的是裸 logits（`--save-logits`）。
而 `compare_golden.py` 吃的是 **jsonl**，所以验收要走 **SGLang 内**那条链：

```bash
# 1) 静态链跑一遍，产出与 golden 同格式的 jsonl（top-32 分布）
python golden_all.py --model $MODEL --tokens 32 --rows 0-0 --concurrent 1 --out /tmp/static_row0.jsonl
# 2) 与 golden 比：value / overlap / flips 三判据
python compare_golden.py --golden golden_bench32k.jsonl --other /tmp/static_row0.jsonl
```

判据（已与 golden 的 `deterministic: False` 对齐）：`--tol 0.15`（实测噪声底约 0.125）、
`--min-overlap 0.6`；argmax flip 只报告、不判失败。

再做两件"证明门槛没定错"的事：

1. **噪声底对照**：`bisect_repeat.sh`（同一 row、两个全新 engine，跑 stock SGLang）→
   用它得到的 maxval 作为 `--tol` 的依据，而不是拍的 0.15。
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
