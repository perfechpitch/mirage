# DeepSeek-V4.1-Flash bs=1 (T=1) 冲延迟 —— Findings

日期：2026-11-24
目标：在 mirage 的 static megakernel 上实现 DSv4.1-Flash-1017 的 **bs=1** 极致推理延迟。

---

## 1. 结论速览

| 项 | 状态 |
|---|---|
| 仓库 remote | `origin` = mirage-project/mirage；新增 `upstream` = perfechpitch/mirage |
| 分支 | `perftest`（性能开关）、`dsv41-bs1`（DSv4.1 + T=1/2/4，已推 upstream） |
| DSv4.1 层代码 | 已在 `dsv41-bs1` 上，编译通过，demo 可 build + launch |
| **T=1/2/4 特化** | ✅ **完成**（`TW` 改造，全部 5 种 attention + MoE） |
| **T=1/2/4 精度** | ✅ **全部逐位等于 T=8 第 0 行** |
| **mHC 精度** | ✅ pre/post/comb ~1e-7，R'/x ~2e-5，4 卡逐位一致 |
| **Engram 精度** | ✅ `ENGRAM=2` R'' relRMS **7.2e-7**；`ENGRAM=1` 7.5e-5 |
| **LM head + argmax** | ✅ logits relRMS 1.2e-4，**argmax 100% 一致** |
| **TP=8 支持** | ✅ **新增**：attention `SA_HL<H>` + MoE I 补到 320，4 个组件全部逐位校验通过，8 卡一致 |
| 端到端 40 层 build | ✅ **314 个 layer module 全部 built + mapped**（209 GB/rank，TP4） |
| 端到端 prefill | ✅ 通过（logits 产出） |
| 端到端 decode | ⚠️ **第 1 步通过，第 2 步挂死**（见 §8.6） |
| 端到端 token 精度 | ❌ 未验证（未跑到出 token） |
| 性能 | 未测（无法独占 GPU，见 §5.1） |

---

## 2. 关键勘察结论

### 2.1 bs=1 的参考实现就在 `mpk` 自己身上

`mpk` tip = `ed771bda`（#838「GLM-5.3-Flash layers Perf for bs1/2/4/64」）已经做完了 T<8 的框架：

- `config.cuh` 的 `static_assert` 接受 `T == 1 || 2 || 4`（以及 8/16/32/64）；
- `config.cuh:31` `constexpr int TW = T < 8 ? 8 : T;` —— tensor core 的 N 最小是 8：
  - **GEMM 宽度相关**（ring/accumulator 列、activation tile、TMA box、scale chunk 行）= `TW`
  - **逐 token 结构**（grid、循环、buffer、reset、send）= `T`
- 激活张量的 TMA box 取 `TW` 行，超出 `T` 的行落在张量外 → 读到 0，天然安全。

#838 的实测（B200，单层，T=8 空转 → T=n，µs）：

| 层 | bs1 | bs2 | bs4 |
|---|---|---|---|
| KDA | 24.7 → 23.9 | 24.4 → 22.7 | 24.8 → 23.9 |
| DSA | 42.1 → 37.4 | 42.1 → 38.9 | 42.1 → 39.7 |
| MoE | 34.4 → 27.2 | 37.1 → 33.2 | 52.4 → 43.7 |

### 2.2 `origin/dsv41-*` 分支的形态（不是整模型 builder）

| 分支 | 内容 | 末次提交 |
|---|---|---|
| `dsv41-csa2` | attention 层（swa/reuse/reindex/full_r1/full_r2），TP4 | 11-09 |
| `dsv41-moe-layer` | + MoE 层（384 experts MXFP4、top-6 sqrtsoftplus、shared e4m3、SwiGLU clamp 10） | 11-09 |
| `dsv41-sgl-layer` | + 挂进 SGLang 的 hook | 11-09 |
| `dsv41-bs8-1119` | 最全：三套 demo + bs1..64 + mHC + Engram + LM head + argmax，34 个提交 | 11-20 |

要点：

- **没有** `register_model_builder("deepseek-v4.1", ...)`，形态是「单层 demo + static megakernel 任务 + SGLang 插件」。
- 全部 `T >= 8`：`config.cuh` 的 assert 只有 8/16/32/64，且 GEMM 宽度直接写 `T`。
- 没有 sm_103 支持（11-05 基线），在 B300 上编译不出来。
- merge-base 是 11-05 `d90383d0`(#819)，不含 #821/#838。

### 2.3 与 `sgl-nvidia-mega` 是两套实现

工作区里 `bench/run_dsv41.sh` + `sgl-nvidia-mega/docs/models/deepseek_v41.md` 那套是
NVIDIA megakernel 插件（`sgl_nvidia_mega.deepseek_v41.runtime`），和 mirage 的 mpk static megakernel 完全独立。

---

## 3. 仓库操作记录

```
origin    git@github.com:mirage-project/mirage.git
upstream  git@github.com:perfechpitch/mirage.git      # 新增
```

git 身份：仓库局部 `user.name = Zhiyuan Xu` / `user.email = zhiyuan.xu@perfectpitch-ai.com`。
`.gitignore` 增加 `/demo/qwen3/mirage_*.csv`（profiler 产物）。

| 分支 | 提交 | 说明 |
|---|---|---|
| `perftest` | `9bfb7b27` | B300/sm_103 + SM100 linear/all-reduce 环境变量化调优开关（15 文件，+275/−82） |
| `dsv41-bs1` | `3c866314` | merge `origin/dsv41-bs8-1119` 到 `perftest` 之上 |
| `dsv41-bs1` | `a82ac949` | 恢复 dsv41 的 `runtime.cuh`（`poll_pause`、`TM_W13_SF`），GEMM 宽度 `T`→`TW` |
| `dsv41-bs1` | `5b65ea80` | **DSv4.1 任务在 T = 1, 2, 4 全部打通** |
| `dsv41-bs1` | `2b7031c2` | `engram_gate` 缓冲区用 `if constexpr` 选择（见 §6.3） |
| `dsv41-bs1` | 本次 | SGLang hook 适配版本漂移（`watermark_state` 可选）+ `sgl_generate.py` 加 `--watchdog`；40 层端到端 build 打通（见 §8.4–8.6） |

`perftest` 的内容：`GPU_CC::B300 = 103`；transpiler 里 `== B200` 判断改 `>= B200`；`kernel.py` 生成 `-arch=sm_103a`；
CMake 加 `100;103`；shared mem 227 KB；`MIRAGE_BK` / `MIRAGE_MMA_M` / `MIRAGE_NUM_AB_STAGES` / `MIRAGE_NUM_C_STAGES`
（linear 与 splitk linear 的 tile/流水线）；`MIRAGE_AR_DIMX` / `MIRAGE_NUM_WORKERS`；qwen3 demo 的
`--load-dir` / `--bench-json` / `--linear-cols-per-task` / `--linear-grid`；decode 吞吐分母改为计时区内步数
（原来把 prefill token 算进 decode，短 prompt 高估约 13%）。

---

## 4. 合并与冲突（重要教训）

合并 `dsv41-bs8-1119` 只有 7 个文件冲突，但**第一版解决方式错了**：

> `git checkout --ours <file>` 会连同该文件里**本来能自动合并**的对侧改动一起丢掉，不只是冲突块。

后果：`runtime.cuh` 取了 mpk 侧，丢掉了 dsv41 的
`poll_pause()` / `STATIC_CNT_POLL_NS` / `STATIC_POLL_NS` / `TM_W13_SF`，
而 dsv41 的 `act_quant.cuh`、`sum_store.cuh`、`gemm_fp8_tile.cuh` 都在调用它们 →
`nvcc` 报 13 个 `identifier ... is undefined`。

**正确做法**：以 dsv41 版本为底，只把「GEMM 宽度」的 `T` 换成 `TW`（`a82ac949`）。

| 文件 | 取舍 | 理由 |
|---|---|---|
| `config.cuh` | mpk（`TW`） | 只有 T<8 的 assert 和注释差异，mpk 侧是超集 |
| `runtime.cuh` | **dsv41 为底 + `TW`** | 保留 `poll_pause` / `TM_W13_SF` |
| `moe_experts.cuh` | mpk | 新版 `build_table` 用前缀和 + 直接查 slot |
| `sum_quant_send.cuh` | mpk | 只差 `static_assert` 的形式 |
| `static_schedule.py` | mpk | 保留 `ar_twoshot` / `reset_sms` / `schedules` |
| `host.cuh` | 两边合一 | 保留 dsv41 的 `box_rows`，默认值改 `TW` |
| `allreduce_send.cuh` | 两边合一 | 同时保留 `AR_TWOSHOT` 与 `TO_T` |

---

## 5. 精度校验

### 5.1 环境约束与 GPU 选择

8 × B300（275 GB/卡）。各卡被其他用户的作业占用的显存与算力：

| GPU | 已用 | 可用 | 算力 |
|---|---|---|---|
| 0 | 60.7 GB | 213 GB | 0% |
| 1 | 54.5 GB | 220 GB | 100% |
| 2 | 54.4 GB | 220 GB | 100% |
| **3** | **210.5 GB** | **63.7 GB** | 100% |
| 4–7 | 37–40 GB | 235–238 GB | 100% |

**TP=4 每卡权重需求 ≈ 118.8 GiB**（475.2 GiB 总量 ÷ 4，含 Engram 表；部分张量是复制的，实际略有出入）。GPU 3 只剩 63.7 GB，装不下 →
**校验用 GPU 0,1,2,4**（不触碰他人进程）。

权重总量按 checkpoint 实际 dtype 统计（`I8`=1、`F8_E8M0`=1、`BF16`=2）：

| 组成 | 大小 |
|---|---|
| routed experts（384 × 40 层，MXFP4 + e8m0） | 268.95 GiB |
| Engram 表（L1/L14） | 189.13 GiB |
| 40 层 attention（wq_a/b、wo_a/b、wkv、compressor、indexer） | 4.78 GiB |
| shared expert | 1.32 GiB |
| embed + head + mTP/aligner | 10.76 GiB |
| **合计** | **475.2 GiB** |

于是 **TP=4 = 118.8 GiB/卡，TP=8 = 59.4 GiB/卡**。TP=8 时连当初 GPU 3 那 63.7 GiB 都勉强够。

精度校验不受算力争用影响（只有计时失真），所以校验期间**不看性能数字**。

### 5.2 TP=4 与 TP=8 都支持了（TP=8 是新加的）

原来 `sparse_attn.cuh` 把每任务 16 个 head 写死（`SA_HPT = 16`，即 MMA 的 M），并有 `H % SA_HPT == 0`，
所以 TP=8（H=8）编译不过。现在引入 `SA_HL<H> = min(H, 16)`：

```cpp
template <int H>
constexpr int SA_HL = H < SA_HPT ? H : SA_HPT;   // 每任务真正有用的 head 数
```

**做法**：MMA tile 仍保持 16 行，但
- tile 里 `[SA_HL, 16)` 这些行的 **q 加载为 0**（不读，避免读到下一个 token 的 q）
- 这些行的 **输出丢弃**（不写 `o_q`、不写 scale chunk）
- `sink[]` 对这些行退化成读 head `SA_HL-1`（保证不越界）
- grid 从 `(T, H/16, ...)` 变为 `(T, H/HL, ...)`；split 的 partials 仍按 `[T][G][NS][16][D]` 分配（一个 tile 一组）

代价几乎为零：bs=1 时 16 行 MMA 只跑到 ~0.3 TFLOP/s，浪费一半算力完全感觉不到；
换来的是**所有 16 行的 tile / stride / buffer 布局一行都不用改**。

MoE 侧：`2304 / 8 = 288` 不是 64 的倍数（`swiglu_moe_experts` 要求 `I % 64 == 0`，
`gemm_fp8_tile` 的共享专家要求 `2 I % 128 == 0`）。所以 `local_inter(tp)` 把 I 补到 320：

| TP | I | IP（补齐后） | W13 items |
|---|---|---|---|
| 1 | 2304 | 2304 | 36 |
| 2 | 1152 | 1152 | 18 |
| 4 | 576 | 576 | 9 |
| **8** | **288** | **320** | **5** |

补齐的行**权重为 0、scale 为 2^0**（e8m0 的 127），于是补出来的 gate/up 输出精确为 0，
`silu(0) * 0 = 0`，fp32 累加器加它是个 no-op —— **逐位不变**。
这和 SGLang 官方 `mxfp4_flashinfer_trtllm_moe.py::_pad_intermediate_size` 用 `(I+127)//128*128`
是同一套做法，只是 mirage 的 item 粒度是 64 行，所以补到 320 比补到 384 省 13 个百分点的浪费。

### 5.3 TP=8 的实测精度（`mhc_check.py --gpus 8`）

| 层 | 指标 | TP8 |
|---|---|---|
| L3 attn | R' / x relRMS | **2.31e-05 / 1.94e-06** |
| L3 MoE | R' relRMS | **3.45e-05** |
| L0 Engram(=2) | R' / R'' relRMS | **2.44e-05 / 4.33e-06**，3 次 relaunch 稳定 |
| L39 head | logits relRMS / **argmax 一致率** | **1.14e-04 / 1.000** |

**8 张卡逐位一致**（每项都验了 `R' the same bits on the other GPUs: [True ×7]`）。
L39 的 argmax 100% 一致 —— 这直接决定 TP=8 下贪心解码能否产出与官方相同的 token。

TP=4 回归验证（同一份共享代码）：

| 层 | 改动前（findings） | 改动后 |
|---|---|---|
| L3 attn R' | 1.72e-05 | **1.72e-05** |
| L3 MoE R' | 2.62e-05 | **2.62e-05** |

数字逐位相同 ⇒ **无回归**。

### 5.3b 为什么 bs=1 要用 TP 而不是 DP+EP（量化依据）

先分清两个场景：**bs=1 延迟**用 TP，**大 batch 吞吐**用 DP+EP。下面只算 bs=1。

bs=1 解码一个 token、每层每卡必须从 HBM 读的字节（只读 top-6 命中的专家）：

| 组件 | TP4 | **TP8** | DP+EP8（最慢那张卡） |
|---|---|---|---|
| attention | 30.6 MiB | **15.3** | 15.3 |
| shared expert | 8.5 | **4.2** | 4.2 |
| routed（6 专家） | 27.9 | **15.9** | **35.9**（摊到 2 个专家 × 全宽） |
| router gate（每卡复制） | 3.8 | 3.8 | 3.8 |
| **每层合计** | 67 | **37** | 57 |
| **40 层 ÷ 7 TB/s** | 412 µs | **226 µs** | 346 µs |

四条理由：

1. **TP 完美均分，EP 不均。** TP 下 6 个专家**每个**都被切 8 份，人人平等；
   EP 下 6 个专家散到 8 个 rank，必有 rank 摊到 2 个、有 rank 摊到 0 个 —— 延迟由**最慢的那张**决定，
   期望 max ≈ 2，直接比 TP8 慢 2.4×。
2. **通信形态。** TP 只要 1 次 all-reduce（10 KB 的 hidden，可走 NVLS multicast）；
   EP 要 dispatch + combine 两次 all-to-all，且 bs=1 时没有任何 batch 可以摊薄。
3. **DP 在并发 = 1 时无意义**（收益是把多个请求分给多卡）。
4. **KV 在 TP 组内是复制的**（MLA，`num_key_value_heads = 1`），
   所以 bs=1 下 DP attention 也拿不到"每卡 KV 变小"的好处。

反过来，大 batch 时 DP+EP 明显更优：EP 是"少量专家 × 全宽"，每个专家上的 token 多了 GEMM 才有密度；
而 TP 切 intermediate 会**引入 padding 浪费**（TP4: 576→640 浪费 11%，TP8: 288→320 浪费 11%）。
这正是 SGLang 在我们日志里给的那条警告的含义：

> flashinfer_mxfp4 MoE padded the local intermediate size from 576 to 640 ...
> **benchmark it against a TP/EP configuration that avoids padding**

这个 trade-off 在 bs=1 时翻转成 TP 占优。

### 5.4 TP=4 的 head 数约束（历史记录）

原 `sparse_attn.cuh` 硬编码每任务 16 个 head：

```cpp
constexpr int SA_HPT = 16;      // heads per task: the MMA's M
```

并有断言 `H % SA_HPT == 0`。模型共 64 head：

| TP | 每卡 head 数 H | 当时的结果 |
|---|---|---|
| 1 | 64 | ✅ |
| 2 | 32 | ✅ |
| **4** | **16** | ✅ **当时选它** |
| 8 | 8 | ❌ 当时 `H % 16 != 0`，编译失败（现已支持，见 5.2） |

官方 checkpoint 默认 TP=8，mirage 侧起初走 TP=4。

### 5.5 Attention：5 个变体 × T=1/2/4 —— 全部逐位等于 T=8

覆盖 `swa`(L0)、`full_r2`(L2)、`full_r1`(L20)、`reuse`(L3)、`reindex`(L24)。

判定方法（不需要官方 golden）：`random_inputs` 给**每个张量独立的 generator**（按名称/序号加种子），
保证第 t 行的输入在所有 T 上都相同。于是：

> T=n 的第 t 行输出必须等于 T=8 的第 t 行输出。

**踩过的坑**：最初用单个共享 `Generator`，`torch.randn(T, HID)` 消耗的随机数正比于 T，
导致后续张量在 T=1 与 T=8 下拿到完全不同的值，attention 误报 `relRMS 1.347e+00`。
修法：

```python
calls = [0]
def gen():
    calls[0] += 1
    return torch.Generator().manual_seed(seed + 1 + calls[0])
rnd = lambda *shape: torch.randint(0, 256, shape, generator=gen(), dtype=torch.uint8).to(dev)
```

修好后 5 个变体 × T∈{1,2,4} **全部逐位相同**。

4 卡 TP=4 一致性（层内 all-reduce 正确）：

```
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python demo/deepseek_v4_1_csa2/demo.py --layer 3 --tp 4 --gpus 4 --reps 5
y bit-identical on the 4 GPUs
y after 5 launches: bit-identical to the first launch's on the 4 GPUs
position 4096: median span 35.7 us over 5 launches
```

### 5.6 MoE：T=1/2/4 —— 逐位等于 T=8 第 0 行

L3：fp64 分块校验 relRMS **6.4e-8**；T=1/2/4 各行与 T=8 对应行逐位相同。

### 5.7 mHC：attn L3 / MoE L3

```
mhc_check.py --kind attn --layer 3
  pre relRMS 2.31e-07  post 2.40e-07  comb 9.51e-08
  R' relRMS 1.72e-05 (equal bits 0.9999)   x relRMS 1.83e-05
  R' the same bits on the other GPUs: [True, True, True]

mhc_check.py --kind moe --layer 3
  pre relRMS 1.04e-07  post 4.33e-07  comb 7.44e-08
  R' relRMS 2.62e-05 (equal bits 0.9997)   x relRMS 5.70e-05 (equal bits 0.9999)
  R' the same bits on the other GPUs: [True, True, True]
```

### 5.8 Engram：两种模式都过

```
mhc_check.py --kind moe --layer 0 --engram --no-norm     # DSV41_ENGRAM=1
  Engram gate: min 0.158 max 0.805
  R'' relRMS 7.53e-05, equal bits 0.99897, max |d| 1.562e-02
  the same bits on every GPU True;  3 次 relaunch 全部 True

mhc_check.py --kind moe --layer 0 --engram2 --no-norm    # DSV41_ENGRAM=2
  Engram2 gate: min 0.193 max 0.832
  R'' relRMS 7.19e-07, equal bits 0.99996
  the same bits on every GPU True;  relaunch 5.23e-06 / 1.25e-08 / 6.35e-07
```

**`ENGRAM=2` 比 `=1` 准两个数量级**（7.2e-7 vs 7.5e-5）：kv 在 attention kernel 里以 fp32 累加算出，
而复用 SGLang 的 gather 路径要经过 bf16 往返。这也是 `README_bs8.md` 推荐的配置。

### 5.9 LM head + argmax：L39

```
mhc_check.py --kind moe --layer 39 --head
  pre relRMS 5.65e-08  post 3.37e-07  comb 6.60e-08
  R' relRMS 3.81e-06   x relRMS 1.79e-05
  head: logits relRMS 1.18e-04, equal 0.9986, argmax equal 1.000
  the same on every GPU True
```

argmax 100% 一致 —— 这直接决定贪心解码能否产出相同 token。

---

## 6. T=1/2/4 实现要点

### 6.1 改动清单（`5b65ea80`）

| 文件 | 核心改动 |
|---|---|
| `runtime.cuh` | `IDESC_F8` 的 N 用 `TW/8`；`float v2[TW]`；`tmem_ld_cols_nw<TW>`；累加器列偏移 `TW * a` |
| `gemm_fp8_tile.cuh` | `FP8_STAGE_TX` 按 `TW`；`acc = tb + TW*a + ACC_STAGES*TW*iss`；`drain_acc` 的 `float v[TW]`；fused act 循环 `32*nst*T` |
| `act_quant.cuh` | 放开 `T ∈ {1,2,4}`；`if constexpr (TW <= 32)` 分支 |
| `sum_store.cuh` | `for (int t = threadIdx.x >> 5; t < T; t += 8)` —— 修掉 `T/8 == 0` 时整个 all-reduce 被跳过 |
| `swiglu_moe_experts.cuh` | `TM_HQ` / `TM_ZQ` 按 `TW`；`BYTES = OFF_ACC + 128*TW*4`；W13 drain 把 `t >= T` 的行写 0 |

### 6.2 CUDA misaligned address（MoE，T=1）

`compute-sanitizer --tool memcheck -lineinfo` 定位到 `issue_w2_job` 的 `mma_mx`：
TMEM 累加器地址 `tb + T * a + ...` 在 `T < 8` 时算出未对齐的列偏移。改成 `TW * a` 即修复。

### 6.3 `engram_gate.cuh` 的槽位 bug（`2b7031c2`）

原代码用普通三元运算符选可选缓冲区：

```cpp
TEN ? buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 3>()) : nullptr
```

**普通 `?:` 的两个分支都会被实例化**，所以 `TEN=false` 的构建仍然要求第 4 个槽位 →
`static_assert` 失败：`the node has fewer slots`。`NX` 的 4/5/6 槽位同理。
改成 `if constexpr` 后只声明真正读取的槽位。

### 6.4 关键不变量

```
TW = T < 8 ? 8 : T        // tensor core 的 N 最小 8
```

- **用 `TW`**：GEMM ring、TMEM 累加器列、TMA box 行、activation tile、scale chunk 行、smem stage 字节
- **用 `T`**：grid 的 token 维、逐 token 循环、令牌 buffer 大小、reset 计数、send 计数
- 越界行（`t ∈ [T, TW)`）必须**显式写 0**，否则 `quant8` 会读到未初始化数据

---

## 7. Engram 的 TP 切分（两套独立切分，别混）

### 7.1 表本身（`embed.weight`）—— 按行切

模型配置：`engram_num_embeddings = [384006168, 384016682]`，`engram_head_dim = 256`，fp8 e4m3 +
每 32 列一个 e8m0 scale。两个 Engram 层（L1、L14）各占 **91.56 GiB + 2.86 GiB scale**，合计 188.8 GiB。

官方 `ParallelEngramEmbedding`（`inference/model.py:296`）：

```python
self.part_num_embeddings = (num_embeddings + world_size - 1) // world_size
self.vocab_start_idx = rank * self.part_num_embeddings
# 越界行 mask 掉置 0，然后 dist.all_reduce(values)
```

SGLang 的 `shard_range`（`srt/layers/engram.py:1230`）优先按**哈希列边界**切：

```python
k = cols_per_rank(len(column_bounds) - 1, tp_size)
if k is not None:
    return column_bounds[tp_rank * k], column_bounds[(tp_rank + 1) * k]
```

24 个哈希列 / 4 = 6，整除，所以 TP4 每卡拥有 6 个完整的 per-head 子表。
**每行恰好一个 rank 拥有** → 零填充 + all-reduce 的结果与切分方式无关（逐位不变）。

### 7.2 `wkv` 投影 —— 按输出行切，K 全量

`wkv = Linear(24*256=6144, 5120*(4+1)=25600)`，fp8 存储，约 **157 MB**。

```python
n = wkv.shape[0] // tp                      # 25600 / 4 = 6400
rows = slice(rank * n, rank * n + n)
```

为什么 K 不能切：gate 要算 `mean(key²)` 的归一化点积，key 必须是完整 6144 维。
所以每卡算 6400 行输出，再用 `allreduce_send` 汇聚成 `[tp·T, W]` 的完整 kv 给 gate 用。

### 7.3 mirage 里的实现位置

| 组件 | 位置 |
|---|---|
| 表查找 kernel | `tasks/engram_rows.cuh`（`DSV41_EG_ROWS=1`，multicast 广播） |
| 门控 kernel | `tasks/engram_gate.cuh` |
| 图节点 | `demo/deepseek_v4_1_moe/demo.py:149` `engram_kv_nodes` / `:175` `engram_nodes` |
| 表接入 | `demo/deepseek_v4_1_sglang/dsv41_static_layer.py:627` `engram_rows_tables` |
| kv 缓冲区 | `:604` `engram_kv_buffer` / `:617` `engram_emb_buffer` |

两种装配模式：

| 模式 | kv 计算位置 | 说明 |
|---|---|---|
| `ENGRAM=1` | MoE kernel 内 | 查表 + wkv + gather + gate 全在 L0/L13 的 MoE kernel |
| `ENGRAM=2` | 前一层 attention kernel 内 | 用 sparse_attn 的空闲 SM；MoE kernel 只做 gate |

`ENGRAM=2` + `EG_ROWS=1` 时连表查找都进 kernel，SGLang 侧的 gather 与 all-reduce 完全省掉。

`engram_rows_tables` 直接复用 SGLang 的切分，两种布局都支持：PEER_READ（单 NVLink 域默认，
通过 VMM 直接读任意 peer 的行，无 all-reduce）与普通分片（本卡 fp8 行 + e8m0 scale，行区间全 TP 组 all-gather）。

---

## 8. 端到端串联：各积木已验证，串联受阻

### 8.1 已就绪的积木

`demo/deepseek_v4_1_sglang/dsv41_static_layer.py` 提供了完整装配层：
`StaticAttn`、`StaticMoE`、`StaticHead`、`ExchangePool`、`Sources`、`StepTables`，支持 mHC、Engram、LM head。
`dsv41_static_hook.py` 通过 `boot/sitecustomize.py` 在 SGLang 进程里 patch 各层。

`bench/README_bs8.md` 给了完整的复用配方：

```
DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_MHC=1
DSV41_ENGRAM=2 DSV41_HEAD=2 DSV41_EG_KV14=1 DSV41_EG_ROWS=1
DSV41_HEAD_ARGMAX=1 DSV41_FILL_FUSED=1 DSV41_SHARE_MODULES=1
DSV41_STATIC_ROWS=1,2,4 DSV41_STATIC_T=16,32,64
```

### 8.2 SGLang hook 路线受阻（三个环境问题）

1. **`ninja` 不在 PATH** —— 某次 JIT 构建因 `FileNotFoundError: ninja` 失败并留下中间状态；
   之后每次启动 4 个 rank 都卡在 `flashinfer/jit/core.py:377 build_and_load`
   的 `filelock._core/_acquire.py:177 _poll_until_acquired`（由 SIGABRT 的 Python traceback 确认）。
   单进程单独构建该模块 **0.57 s**（已缓存），所以不是编译慢，是锁被占住。
2. **显存不足** —— Engram 表放 GPU 时权重+表占 190/275 GB，
   `mem_fraction_static=0.7` 分不到 KV cache（报错要求 > 0.713）。改 `placement=cpu` 又卡在 pinned-host 的 pre-fault。
3. **版本漂移** —— 当前 SGLang `2be81e0ddc`，比 `README_bs8.md` 要求的 `240f2a0a64` **领先 1717 个提交**。

### 8.3 绕开方案

mirage 自带的 `mhc_check.py` 不依赖 SGLang，用官方 fp32 公式在 CPU 上对照，
且支持 `--engram` / `--engram2` / `--head` —— 已用它覆盖全部积木（§5.5–5.7）。
缺的只是「40 层串联 + 层间状态传递」这一步。

### 8.4 上述三个阻塞已全部解决

| 问题 | 根因 | 解决 |
|---|---|---|
| FlashInfer 锁 | 首次启动时 rank 3 持锁跑 `ninja`/`nvcc` 编译 50 个 `sm103a` JIT kernel；SGLang 默认 300 s watchdog 在编译期间就把进程杀了 | 编译产物已缓存（`~/.cache/sglang/.cache/flashinfer`、`~/.cache/sglang/jit/sm103a`）；`sgl_generate.py` 新增 `--watchdog` |
| 显存不足 | GPU 0 有他人 53 GiB 占用 | 换 GPU **1, 4, 5, 6**（各 >230 GB 空闲）+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--mem-fraction 0.90` |
| 版本漂移 | `ModelRunner.sample` 不再传 `watermark_state` | `recorded_ids` 改成 `watermark_state=None` 可选参数，回退读 `si.watermark` |

`--watchdog` 与 `--engram-config` 已加进 `sgl_generate.py`（`5b65ea80` 之后的未提交改动，见 §3）。
坑记录：`SGLANG_DSV4_KV_LAYOUT=v41_new` 必须设，否则 `KVLayout.V4` assert 直接挂。

### 8.5 端到端 build 成功：314 个 module、209 GB/rank

```
DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_MHC=1
DSV41_ENGRAM=2 DSV41_EG_KV14=1 DSV41_HEAD=2
```

4 个 rank 各自把 40 层的 attention + MoE + mHC + Engram + head 编译并 map 完成
（日志 `attention L18 (...) ready, 64.8 GB free` × 314），之后成功执行：

- prefill → 产出 logits ✅
- 第 1 个 static decode step ✅

**结论：`StaticAttn` / `StaticMoE` / `StaticHead` / `ExchangePool` 这套装配层在 40 层规模下是能起来的。**

一个副产品观察：

```
head argmax: 0 batches from the kernel's ids, 1 by the sampler
```

即 kernel 内 argmax 这条快速路径**没有命中**，退化到 sampler 自己算 argmax（结果应当一致，但没有拿到 kernel 侧收益）。
`DSV41_HEAD_ARGMAX=1` 的命中条件需要单独排查（`_head_by_logits` / `_ok_rows` / `data_ptr` 对齐）。

### 8.6 decode 第 2 步挂死 —— 我误用了激进 flag

第 1 步通过后，进程不再前进，但**不是崩溃**：

| 观测 | 值 | 判读 |
|---|---|---|
| GPU 利用率 | 5 次采样全 100% | 空转 |
| 4 个 rank 的 CPU | 109–111%，`Rl` | 在跑，不是在睡 |
| 日志 | 7 分钟无新增 | 无进展 |
| Xid 错误 | 无 | 不是硬件 |

典型**计数轮询死循环**：kernel 等一个永远不会到的计数器，而我把
`STATIC_CNT_POLL_NS=0` / `STATIC_POLL_NS=0`（不睡眠轮询）打开了，于是 100% 空转。

**根因判断（我的操作失误）**：我把 `README_bs8.md` 里那套**为 bs=8 实测标定的激进 flag**
一次性全开了——`STATIC_NO_START_BARRIER`、`RESET_PARITY`、`HCP_FAST_END`、PDL、poll ns=0——
而那套配置的标定前提里还带着 `DSV41_STATIC_ROWS=1,2,4` 和 `DSV41_STATIC_T=16,32,64`，这两项我没设。

重跑时改用**最小安全配置**（`PDL False`，只留 `ATTN=all MOE=all MHC=1 ENGRAM=2 EG_KV14=1 HEAD=2`），
已确认 hook 打上、build 正常推进到 L18；随后因 `/dev/shm` 占满被主动停掉。

**教训（顺序问题）**：应当先用最小配置证明「40 层 + 8 个 token 与官方 baseline 逐位相同」，
**再**逐个打开 flag 看单步耗时。我这次把优化 flag 和正确性验证混在一起跑了。

### 8.7 `/dev/shm` 被打满

`/dev/shm` 是 2.0 TB 的 tmpfs。4 个 rank 的 exchange pool + SGLang 的 IPC 共享内存会吃掉它，
跑到一半 `/dev/shm` 接近 100%。杀掉进程后回落到 **1.6 GB**，说明是正常释放的（不是泄漏文件残留）。

后果是其他并发作业也受影响，所以这轮验证主动停止。后续跑之前应先看 `df -h /dev/shm`，
并考虑给 exchange pool 换到别的暂存位置，或缩小 `DSV41_STATIC_ROWS` 的上限。

---

## 9. 下一步

1. **端到端正确性（最高优先）**：最小安全配置 + `DSV41_STATIC_ROWS=1`（单行），
   跑 8 token，与 `/tmp/ref_t8.json` 对照
   （官方 SGLang baseline，8 条同 prompt 贪心解码：`[5, 17209, 15, 23021, 55607, 83224, 271, 37460]`）。
   逐位相同之后才谈性能。
2. **TP=8 端到端**：层代码已验，还差把 `DSV41_STATIC_*` 那套装配层在 TP=8 下拼起来
   （`dsv41_static_layer.py` 的 head / Engram 分片本来就按行切，理论无需改）。
3. **定位 decode 第 2 步挂死**：在最小配置下复现，用 `compute-sanitizer --tool synccheck`
   或给 kernel 的 poll 循环加计数器 dump，找出等的是哪个 tick / 哪个 `SEND::counter`。
4. **`DSV41_HEAD_ARGMAX` 未命中**：查 `_head_by_logits` / `_ok_rows` / `data_ptr` 的对齐条件。
5. **性能特化**：按 T=1 → T=2 → T=4 → T=8 逐档，需要独占 GPU 才能计时。
   TP=8 每卡只需 ~59.4 GiB，8 张卡全都够。
6. **`sparse_attn` 的 T=1 SM 利用率**：grid 是 `(T, H/HL, ...)`，T=1 + TP4 时每个 KV 分片只有
   1×1 = 1 个 SM；TP8 时 H/HL = 1，同样是 1 个 —— 这是 bs=1 最大的剩余收益点
   （#838 的 `dsa_mla_merge` / T=1 absorb 融合是同类做法）。
7. **可选**：写 `make_golden.py`（sglang venv + tilelang）产出官方 cases，补 attention 的绝对精度。
