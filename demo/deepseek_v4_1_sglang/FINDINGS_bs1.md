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
| 端到端 40 层 build | ✅ TP4 314 个 module 全部 built + mapped（209 GB/rank）；**TP8 40 attn + 40 MoE 也全部 built**（缓存命中，见 §8.8） |
| 端到端 prefill | ✅ 通过（logits 产出） |
| 端到端 decode | ❌ **首个真实 step 不返回**（TP4/TP8 都是；单层即复现，见 §8.6 / §8.8.3） |
| CUDA graph 捕获 | ✅ 通过（但捕获只记录不执行，**不能**作为 decode 跑通的证据，见 §8.8.3） |
| 端到端 token 精度 | ❌ 未验证（未跑到出 token） |
| 性能 | 未测（decode 未跑通；bench32k/TPOT 脚手架已就绪，见 §8.8.5） |

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

### 5.3b 为什么 bs=1 要用 TP 而不是 DP+EP / EP2（量化依据）

先分清两个场景：**bs=1 延迟**用 TP，**大 batch 吞吐**才轮到 DP+EP。下面只算 bs=1。

#### TP8 与 "TP8 + EP2" 的确切切分

SGLang 的定义（`runtime_context.py`）：

```python
"moe_tp_size": tp_size // moe_ep_size // moe_dp_size,
```

所以 `tp=8, ep=2` 意味着：

| 组件 | 切分 | 代码依据 |
|---|---|---|
| attention（wq_a/b、wo_a/b、kv、indexer） | **TP = 8** | `parallel_group="attn_tp"` |
| routed MoE | **EP 2 组 × 组内 TP 4** | `moe_tp_size = 8//2//1 = 4` |
| shared expert | **TP = 8**（不降级） | `parallel_group="tp"` → `tp_group`，不是 `moe_tp_group` |
| embed / head / norms | TP = 8 | — |

具体形态：384 个专家切成 2 组（192 个/组），每个专家被组内 4 个 rank 按 intermediate 切 4 份
（`intermediate_size_per_partition = 2304 // 4 = 576`）。
每卡内存 = 192 × 1/4 = **48 个专家当量**，和 TP8 的 384 × 1/8 = 48 完全一样（都是 59.4 GiB/卡）。

两个常被误解的点：

- **shared expert 不会跟着 EP 降级**。它用 `tp_group`（8 卡），不是 `moe_tp_group`（4 卡）。
  只有走 DeepEP/MegaMOE 这类 A2A 后端时才变 `"replicated"`（每卡全量复制）。
- **没有额外的两次 all-to-all**。不带 A2A 后端时，归约就是一次跨 8 卡的普通 all-reduce
  （`if reduce_results and (moe_tp_size > 1 or moe_ep_size > 1): tensor_model_parallel_all_reduce(...)`），
  通信量和 TP8 相同。

#### bs=1 每层每卡从 HBM 读的字节

| 配置 | routed | shared | attn+gate | 合计 | 40 层 ÷ 7 TB/s |
|---|---|---|---|---|---|
| **纯 TP8** | 15.94 | 4.22 | 19.14 | **39.29 MiB** | **235 µs** |
| TP8 + EP2 | 18.30 | 4.22 | 19.14 | **41.66 MiB** | **250 µs** (+6.0%) |
| DP+EP8（最慢卡） | 35.9 | 4.2 | 19.1 | 59.2 | 355 µs |

**EP2 唯一的劣势就是路由不均**：6 个专家随机落到 2 组，慢组专家数服从 `max(X, 6−X)`，`X ~ Bin(6, 0.5)`：

| 慢组拿到 | 概率 | routed | vs TP8 |
|---|---|---|---|
| 3 | **31.2%** | 13.95 MiB | **0.88×（赢）** |
| 4 | 46.9% | 18.59 | 1.17× |
| 5 | 18.8% | 23.24 | 1.46× |
| 6 | 3.1% | 27.89 | **1.75×** |

`E[max] = 3.94`（均值 3.0，**1.3× 不均**）。也就是说：
**EP2 用 69% 概率变慢（最坏 +75%）去换 31% 概率快 12%**。而纯 TP8 的 39.29 是恒定的，零抖动。
对 bs=1 冲延迟而言，抖动来自路由、不可控 —— 所以选纯 TP8。

#### 其余三条理由

1. **TP 完美均分，EP 不均。** TP 下 6 个专家**每个**都被切 8 份；EP 下必有 rank 摊到 2 个、
   有 rank 摊到 0 个，延迟由**最慢那张**决定。
2. **DP 在并发 = 1 时无意义**（收益是把多个请求分给多卡）。
3. **KV 在 TP 组内是复制的**（MLA，`num_key_value_heads = 1`），
   所以 bs=1 下 DP attention 也拿不到"每卡 KV 变小"的好处。

#### 一个必须知道的限制

`overrides.py` 里**所有 A2A 后端都会把 ep_size 强改成 tp_size**：

```python
_A2A_EP_SPANNING_BACKENDS = {"megamoe","deepep","deepep_v2","mooncake","nixl",
                             "ascend_fuseep","flashinfer_nvlink_one_sided",
                             "flashinfer_megamoe","mori","pplx","ascend_ops"}
def _a2a_ep_size(view):
    if view.moe_a2a_backend in _A2A_EP_SPANNING_BACKENDS:
        return {"ep_size": view.tp_size}     # ep=2 会被改成 8
```

所以 **`tp=8, ep=2` 只在 `moe_a2a_backend=none` 时成立**；一旦上 DeepEP/MegaMOE 就自动变成 `ep=8`。

#### 大 batch 时反过来

EP 是"少量专家 × 全宽"，每个专家上的 token 多了 GEMM 才有密度；而 TP 切 intermediate 会
**引入 padding 浪费**（TP4: 576→640，TP8: 288→320，各 11%）。T=8 时两者基本打平
（`E[max] = 26.8/48`，不均降到 11.5%），T=64+ 时 EP 明显更好。这正是 SGLang 那条警告的含义：

> flashinfer_mxfp4 MoE padded the local intermediate size from 576 to 640 ...
> **benchmark it against a TP/EP configuration that avoids padding**

**比 EP2 更值得做的**：EP2 那 31% 的赢面本质是"省掉 TP8 的 padding"。直接把 padding 干掉更好 ——
W13 的 item 从 64 行改成 32 行，288 = 9×32 正好整除，routed 从 15.94 → **14.94 MiB（−6.3%）**，
确定性收益、零抖动、不用 EP。w2 的 K 仍需补到 384（128 行 tile 要求），所以只赚 6.3% 而非 11%。

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

### 8.8 TP=8：40 层装配打通，但首个真实 decode step 挂死（本轮结论）

这一轮把 TP=8 的 40 层端到端**装配**走通了（此前只到层级别校验），并在过程中修掉一个
TP=8 专属 bug；但 **decode 依然走不通**，且已经把根因从“装配/编译”收窄到“kernel 首次真实 launch”。

#### 8.8.1 修掉的 bug：`act_quant` 的 2048 列假设（`50c56461`）

attention 的 `wo_a` 输出宽度是 `G * OL`：**TP4 是 2048，TP8 只有 1024**。
`act_quant_layer` 走“GEMM 的 K parts”那条分支时，一个任务固定 2048 列：

```python
assert K % 2048 == 0                          # 旧
grid = (T, K // 2048, 1)                      # 旧：TP8 下变成 (T, 0, 1)
```

SWA 层不走 wo_b 的 fused 路径，于是 `(T, 0, 1)` 的 grid 直接触发断言，40 层 build 死在 L36 之前。
kernel 侧 `act_quant_parts_task` 的循环步长本来就是 `8 * 256 = 2048`，
所以一个任务只要占**整数个 128 列块**就成立（128 列 = 一个 warp 的 scale line 粒度，
32 列 = 一个 shuffle 组），NC < 2048 时第 128 号以后的线程自然空转。

改法：取 `K` 能整除的最大合法列数（2048/1024/512/256/128），
kernel 的 `static_assert` 从 `NC % 2048 == 0` 放宽到“整数个 128 列块、至多一个 2048 的 pass”。
手写 plan 里 `oa_q` 的 grid 本来就是 `max(1, G*OL // 2048)`，与改动一致。

#### 8.8.2 显存：静态层自己持有一份权重副本

`dsv41_static_layer.py` 的每层都通过 `reader(ckpt)` **从 checkpoint 重新读一遍权重**
（`layer_tensors` / `mhc_tensors` / `engram_kv_tensors`，按它自己的布局：w13 补到 320、
scales 按 128×128 tile 切块），不是复用 SGLang 已加载的 tensor。所以每卡是两份：

| 项 | TP8 每卡 | 说明 |
|---|---|---|
| SGLang 权重 | 50.66 GB | 日志 `Load weight end ... mem usage=50.66 GB` |
| 静态层副本 | ~38.4 GB | 307.2 GB model 权重 / 8；engram 203.1 GB 在 **CPU**，不进 HBM |
| KV（mem-fraction 0.85） | ~177 GB | 挤掉了静态副本的空间 |

用 safetensors header 逐 tensor 统计：**model 307.2 GB + engram 203.1 GB = 510.3 GB**，
engram 走 `--engram-config '{"placement":"cpu"}'` 留在主机侧（rank 进程 host RSS 只有 14.8 GiB 可印证）。

`MEMFRAC=0.85` 时静态层构建到 **L36 OOM**（只剩 0.6 GB）；
`MEMFRAC=0.5` 下 40 层构建全程余量 90–180 GB，稳定通过。

> 因此“静态层建好之后释放 SGLang 那份权重”是**不可行**的：
> hook 只替换 decode 的 `MQALayer.forward` / `DeepseekV2MoE.forward`，
> **prefill 仍然跑 SGLang 原生 kernel**（bench32k 正是 32K prefill 的场景）；
> 且两份是不同布局，不能只改指针。

#### 8.8.3 挂死：现象与证据链

现象：40 层（或单层）装配成功、CUDA graph 捕获成功，**第一个真实 decode step 之后不再前进**，
GPU 100% 占用、日志静止。

| # | 实验 | 结果 | 判读 |
|---|---|---|---|
| 1 | 纯 SGLang（不加载 hook） | 8 token / 7.4 s，答案正常 | 环境、模型、mHC 路径本身没问题 |
| 2 | 40 层全静态 `MHC=1 ENGRAM=2 HEAD=2` | 挂死 | — |
| 3 | `MHC=0 ENGRAM=0 HEAD=0` | 挂死 | 与“mHC 由谁接管”无关 |
| 4 | 仅 L0 静态（`ATTN=0 MOE=0`） | 挂死 | **不是 40 层累积，单层就复现** |
| 5 | 预热过的 Triton 缓存（`hc_combine` 72 条 / `hc_mix_stats` 48 条） | 仍然挂 | **不是编译、不是缓存** |
| 6 | 挂住时另起进程在 GPU0 跑 1024³ matmul | 1.5 s 正常返回 | **GPU 本身健康**，是本进程 context 被占 |
| 7 | `py-spy dump`（8 个 rank） | 全部停在 `cuModuleLoadData` | 宿主被一个不返回的 kernel 阻塞 |

采样栈（`py-spy`，`MHC` 开/关时分别命中不同 kernel）：

```
CompiledKernel._init_handles (triton/compiler/compiler.py:479)     ← cuModuleLoadData
  CompiledKernel.launch_metadata
    JITFunction.run
      hc_combine_norm_mxfp8      ← 或 hc_mix_stats_sinkhorn（MHC=0 时）
        _combine / mix_stats (deepseek_v4_mhc.py)
          HcReadCall.read / HcUpdate.post (deepseek_v4_boundary.py)
```

**关键结论：栈顶是症候，不是病因。** 宿主卡在 `cuModuleLoadData`，
是因为 context 里有一个永不返回的 kernel（静态 megakernel），
下一次需要 context 独占的 CUDA 调用就永久阻塞。
`cuModuleLoadData` 之所以出现在这里，是因为 `MHC` 那条 boundary 路径恰好是
“静态 kernel launch 之后的第一处宿主 CUDA 调用”（`quantize and x.shape[0] <= 128`
才走 `hc_combine_norm_mxfp8`，纯 SGLang 走的是另一个 kernel，所以从没触发过）。

也正因为**捕获只记录、不执行**，CUDA graph 捕获阶段能打印
`first static decode step (L0): in a CUDA graph capture` 却依然在真实执行时挂死 ——
这条日志**不能**当作“decode 跑通”的证据。

已排除项：`DSV41_STATIC_PDL` 默认就是 0；`MHC/ENGRAM/HEAD` 全关无效；
`DSV41_STATIC_MAX_POS` 从 33024 退回默认 2047 无效；Triton 缓存预热无效。

#### 8.8.4 下一步的定位手段（未执行）

问题是“某个 poll / counter 永远不被满足”，与 §8.6 同一类。但 §8.6 归因于激进 flag，
而这一次用的是**最小安全配置**、且单层就复现，所以必须换手段：

1. **脱离 SGLang 的最小复现**：用 `mhc_check.py` 那套（单层、自己造输入、不经 SGLang）
   调一次静态层，确认 kernel 本体能否返回。这是最快的一刀，能把 SGLang 完全排除。
2. **逐节点裁剪**：单层里只留 `act_quant` + 一个 GEMM，再逐个加节点，找出第一个不返回的。
3. **counter dump**：给 poll 循环加计数器打印（或 `compute-sanitizer --tool synccheck`），
   看它等的是哪个 tick / 哪个 `SEND::counter`。
4. 注意 `DSV41_STATIC_ROWS=1` 与 `T=8` 的组合：单层实验里 8 行 step 走的是 T=8 的 build，
   活行只有 1，idle 行按 `rows_variant` 处理，这条路径本身也需要单独确认。

#### 8.8.5 附带产物：bench32k / TPOT 的脚手架

`bench/run_dsv41_tp8_bench32k.sh`（本轮新增）：起一个**长跑 SGLang 服务**，
decode 走静态 megakernel，由 LCA bench32k client 打点。

- **bench32k 指 32K 上下文**（`data/long-code-arena/bench32k/*.jsonl`，ISL 22k–33k），**不是 bs=32**。
- **TPOT = `step_p50_ms`**（客户端 SSE 相邻 chunk 间隔；静态路径是纯 decode、无 DSPARK，
  一个 step 就是一个 token，所以它直接就是 TPOT，不需要按 accept length 折算）。
- 因为走纯 TP8（`--ep-size 1`），与 NVIDIA megakernel 那套 `tp8 = --tp 8 --ep-size 2` + DSPARK
  不是同一口径，跨臂比较时要注明。
- 该脚本目前**跑不出版本号内的 TPOT**：请求会死在第一个 decode step（见 8.8.3）。

参考口径（NVIDIA megakernel + DSPARK，`bench32k` medium_context，n=20，osl=512）：
`step_p50 3.006 ms`、`tps_user_p50 1445.9`、`ttft_p50 374.2 ms`。

#### 8.8.6 运维教训

- **不要用宽 `pkill -f` 清共享机器上的 GPU 进程**。`pkill -f "sglang::scheduler"`
  会命中同账号下其他会话的 sglang。正确做法：先 `nvidia-smi --query-compute-apps=pid`
  取 PID，再按**自己启动的 PID** 白名单 kill；kill 循环里应打印每一行再杀。
- 启动前确认 `/dev/shm` 与每卡 `memory.free`（`run_dsv41_tp8_40l.sh` 已有检查）。
- 40 层 TP8 的 `.so` 已全部在 `build/sgl_static_dsv41/cache/`，重启命中缓存，
  构建阶段不再是瓶颈（主要耗时是权重加载 ~108 s）。

---

## 9. 下一步

> 本轮之后的状态：TP=8 的 40 层**装配**已打通（见 §8.8），
> decode 仍卡在“首个真实 step 不返回”，且已收窄到 kernel 本体、单层可复现。

1. **先做脱离 SGLang 的最小复现（最高优先）**：用 `mhc_check.py`（不经 SGLang、自造输入）
   跑一次单层静态层，确认 kernel 本体能否返回。这一步能把 SGLang 完全排除，
   决定后续是查 kernel 还是查接线。
2. **再逐节点裁剪定位**：在单层里只留 `act_quant` + 一个 GEMM，逐个加节点，
   找出第一个不返回的；配合 poll 循环计数器 dump 或 `compute-sanitizer --tool synccheck`。
   注意单层实验里 8 行 step 走 T=8 的 build（活跃行 1），`DSV41_STATIC_ROWS` 那条路径要单独确认。
3. **端到端正确性（挂死解决后最高优先）**：最小安全配置 + `DSV41_STATIC_ROWS=1`，
   跑 8 token，与官方 SGLang baseline 的贪心 token
   （`[5, 17209, 15, 23021, 55607, 83224, 271, 37460]`）逐位对照。逐位相同之后才谈性能。
4. **`DSV41_HEAD_ARGMAX` 未命中**：查 `_head_by_logits` / `_ok_rows` / `data_ptr` 的对齐条件。
5. **性能特化**：按 T=1 → T=2 → T=4 → T=8 逐档，需要独占 GPU 才能计时。
   TP=8 每卡只需 ~59.4 GiB，8 张卡全都够。
6. **`sparse_attn` 的 T=1 SM 利用率**：grid 是 `(T, H/HL, ...)`，T=1 + TP4 时每个 KV 分片只有
   1×1 = 1 个 SM；TP8 时 H/HL = 1，同样是 1 个 —— 这是 bs=1 最大的剩余收益点
   （#838 的 `dsa_mla_merge` / T=1 absorb 融合是同类做法）。
7. **可选**：写 `make_golden.py`（sglang venv + tilelang）产出官方 cases，补 attention 的绝对精度。
