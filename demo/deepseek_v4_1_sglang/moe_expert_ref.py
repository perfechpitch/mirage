"""Expert-level MoE reference: the checkpoint's fp4 experts, in fp64, scored against both SGLang and the kernel.

Two modes.

  (a) vs SGLang's own FFN output (`L{LL}_mpost_x`): the end-to-end fidelity of this layer's FFN.
  (b) `--from-chain DIR`: the KERNEL's own y, deconvolved out of its `streams_out` with `y = (R' - comb^T R)/post`
      (least squares over the four mHC streams), compared to the checkpoint math on the *identical* x and routing.
      This is the one that isolates the kernel: no trajectory, no reference nondeterminism, no gate difference.

Why the fp64 reference needs `act_quant`: the kernel does not feed fp64 x to the experts. Its launch plan runs
`act_quant` twice (grids `act_quant#9`, `act_quant#15`): x -> fp8 e4m3 with one power-of-two scale per 32 columns
(`tasks/act_quant.cuh`, ue8m0, satfinite at +-448), and after the SwiGLU the product -> fp8 e4m3 the same way, into w2.
e4m3 has three mantissa bits, so skipping that quantization makes a *correct* kernel look ~3% wrong. `quant_probe.py`
measures exactly that.

Weight decoding (this is what made the older `moe_ref.py` disagree at cos 0.23): the checkpoint carries TWO scale
grids. The routed experts are per-row-per-32-inputs, `[out, in/32]`; the shared expert and the attention linears are
true 32x32 blocks, `[out/32, in/32]`. Applying the per-row formula to a block grid misaligns every scale.

    python moe_expert_ref.py --layer 1                             # vs SGLang
    python moe_expert_ref.py --layer 1 --from-chain /tmp/chain_fresh  # vs the kernel
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

CK = "/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017"
GOLD = "/tmp/dec_coef8/layers/step1"
H, HC, INTER = 5120, 4, 2304
SWIGLU = 10.0
EPS = 1e-20
K = 6

# float4 e2m1: c = s e e m  ->  e == 0 ? 0.5*m : (1 + 0.5*m) * 2^(e-1)
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
LUT = np.concatenate([E2M1, -E2M1])


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


class CK_:
    def __init__(self):
        self.idx = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]

    def __call__(self, n):
        with safe_open(os.path.join(CK, self.idx[n]), "pt") as sf:
            return sf.get_tensor(n)


CK_ = CK_()


def scale_of(t):
    """e8m0: a power of two, biased by 127."""
    assert t.dtype == torch.float8_e8m0fnu, t.dtype
    return torch.pow(2.0, (t.view(torch.uint8).float() - 127.0))


def _expand(sc, out, inn):
    s = scale_of(sc).double().numpy()
    if s.shape == (out, inn // 32):
        return np.repeat(s, 32, axis=1)
    if s.shape == (out // 32, inn // 32):
        return np.repeat(np.repeat(s, 32, axis=0), 32, axis=1)
    raise ValueError(f"unexpected scale shape {s.shape} for ({out}, {inn})")


def _rne_e4m3(v):
    """float32 -> e4m3 (1 sign, 4 exp, 3 mantissa), RNE, satfinite at +-448, as cvt_e4m3_nv."""
    a = np.abs(v).astype(np.float64)
    s = np.sign(v)
    out = np.zeros_like(a)
    live = a > 0
    if live.any():
        e = np.clip(np.floor(np.log2(np.maximum(a, 1e-30))).astype(np.int64), -6, 8)
        scale = np.exp2(e)
        q = np.rint((a / scale) * 8.0) / 8.0
        up = q >= 2.0
        q = np.where(up, q / 2.0, q)
        scale = np.where(up, scale * 2.0, scale)
        out = s * q * scale
    sub = live & (a < 2.0 ** -6)
    if sub.any():
        out[sub] = s[sub] * np.rint(a[sub] / 2.0 ** -9) * 2.0 ** -9
    return np.clip(out, -448.0, 448.0)


def act_quant(x, kbits=False):
    """The kernel's quantization of a row: one power-of-two scale per 32 columns, then e4m3 (round trip).

    `kbits` reproduces tasks/act_quant.cuh's exponent bit for bit: `ceil_log2_bits(fmaxf(amax, 1e-4f) * (1.0f / 448.0f))`
    -- the multiply by the float32 reciprocal of 448, not a true divide. The two differ in the last bit of the product,
    and `ceil(log2(.))` turns that into a whole exponent (a factor of 2 on all 32 columns) whenever the true ratio sits on
    a power of two. That is the one place a mathematically-equivalent implementation can disagree by 2x on a group, and
    one bad group is worth 1.7-4.6% of an expert's output (see FINDINGS 12.16)."""
    v = np.asarray(x, np.float64).reshape(-1, 32)
    amax = np.maximum(np.abs(v).max(1), 1e-4)
    if kbits:
        # fmaxf(amax, 1e-4f) is exact (bf16 inputs), then the float32 product, then ceil(log2) from its bits
        t = (amax.astype(np.float32) * np.float32(1.0 / 448.0)).astype(np.float32).astype(np.float64)
        m, e2 = np.frexp(t)                      # t = m * 2**e2, m in [0.5, 1)
        ex = np.where(m == 0.5, e2 - 1.0, e2)    # exact power of two -> that exponent, else round up
    else:
        ex = np.ceil(np.log2(amax / 448.0))
    return (_rne_e4m3(v / np.exp2(ex)[:, None]) * np.exp2(ex)[:, None]).reshape(-1)


def unpack_fp4(w, sc):
    o, half = w.shape
    b = w.view(torch.uint8).numpy()
    v = np.empty((o, half * 2), np.float64)
    v[:, 0::2] = LUT[b & 0xF]          # little nibble first: element 2i is the low nibble
    v[:, 1::2] = LUT[(b >> 4) & 0xF]
    return v * _expand(sc, o, half * 2)


def unpack_fp8(w, sc):
    return w.double().numpy() * _expand(sc, *w.shape)


def expert_raw(ck, L, e, x, quant_x=False, quant_act=False, kbits=False):
    """One expert's SwiGLU, fp64. `e=None` is the shared expert (fp8 weights, no routing weight).

    `quant_x` / `quant_act` reproduce the kernel's two `act_quant` stages; see the module docstring."""
    q = f"layers.{L}.ffn." + ("shared_experts." if e is None else f"experts.{e}.")
    w1, w3, w2 = ck(q + "w1.weight"), ck(q + "w3.weight"), ck(q + "w2.weight")
    U = unpack_fp4 if w1.dtype == torch.int8 else unpack_fp8
    xi = act_quant(x, kbits) if quant_x else np.asarray(x, np.float64)
    g = xi @ U(w1, ck(q + "w1.scale")).T
    u = xi @ U(w3, ck(q + "w3.scale")).T
    # model.py Expert.forward: the GEMM output is bf16, then up is clamped both sides and gate from above only
    g = torch.from_numpy(g).float().bfloat16().float().numpy()
    u = torch.from_numpy(u).float().bfloat16().float().numpy()
    u = np.clip(u, -SWIGLU, SWIGLU)
    g = np.minimum(g, SWIGLU)
    a = (g / (1.0 + np.exp(-g))) * u
    # tasks/act_quant.cuh act_quant_swiglu_task: `y[k] = bf16r(g / (1.0f + expf(-g)) * u)`. The SwiGLU product is
    # rounded to bf16 BEFORE the fp8 quantization. Omitting it perturbs each element by ~0.2%, which flips ~13% of
    # them across an fp8 boundary (steps are 3-6%) and costs ~1.5% of an expert's output -- the whole MoE gap.
    a = torch.from_numpy(a).float().bfloat16().float().numpy().astype(np.float64)
    if quant_act:
        a = act_quant(a, kbits)
    return a @ U(w2, ck(q + "w2.scale")).T


def gate_of(x, gw, gb):
    """Top-6 with norm_topk_prob -> route_scale (validated bit-for-bit against the kernel's pairs, FINDINGS 12.11)."""
    sc = np.sqrt(np.log1p(np.exp(x @ gw.T)))
    order = np.argsort(-(sc + gb))
    top = order[:K]
    return top, sc[top] / sc[top].sum() * 1.5


def kernel_y_of(so, rin, post, comb):
    z = so - comb.T @ rin
    return (post @ z) / max(float(post @ post), 1e-30)


def load_x_and_residual(d, L):
    def bf(n):
        t = torch.frombuffer(bytearray(open(os.path.join(d, n), "rb").read()), dtype=torch.bfloat16).float()
        return t.numpy().reshape(-1, HC, H)[0].astype(np.float64)

    x = torch.frombuffer(bytearray(open(os.path.join(d, f"moe{L}_x.bin"), "rb").read()),
                         dtype=torch.bfloat16).float().numpy().reshape(-1)[:H].astype(np.float64)
    coef = np.fromfile(os.path.join(d, f"moe{L}_coef.bin"), np.float32).reshape(-1, 32)[0]
    post, comb = coef[4:8].astype(np.float64), coef[8:24].reshape(HC, HC).astype(np.float64)
    return x, bf(f"attn{L}_streams_out.bin"), post, comb, bf(f"moe{L}_streams_out.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--from-chain", metavar="DIR", default=None)
    ap.add_argument("--quant", action="store_true", help="reproduce the kernel's two act_quant stages")
    a = ap.parse_args()
    L, ck = a.layer, CK_
    p = f"layers.{L}.ffn."
    f32 = lambda n: np.fromfile(os.path.join(GOLD, n), np.float32).astype(np.float64)

    if a.from_chain:
        x, rin, post, comb, so = load_x_and_residual(a.from_chain, L)
        target, label = kernel_y_of(so, rin, post, comb), "the KERNEL's own y"
        print(f"x from {a.from_chain}/moe{L}_x.bin: |x| rms {np.sqrt((x ** 2).mean()):.5f}")
    else:
        pre = f32(f"L{L:02d}_ct_pre.f32").reshape(-1)[:HC]
        rin = f32(f"L{L:02d}_apost.f32").reshape(HC, H)
        y = (pre[:, None] * rin).sum(0)
        g = ck(f"layers.{L}.ffn_norm.weight").double().numpy()
        x = y / np.sqrt((y ** 2).mean() + EPS) * g
        target = f32(f"L{L + 1:02d}_mpost_x.f32")
        label = f"L{L + 1:02d}_mpost_x"
        print(f"x = ffn input: |x| rms {np.sqrt((x ** 2).mean()):.5f}")

    gw = ck(p + "gate.weight").float().numpy().astype(np.float64)
    gb = ck(p + "gate.bias").float().numpy().astype(np.float64)
    eid, wgt = gate_of(x, gw, gb)
    print(f"gate: experts {eid.tolist()}  weights {np.round(wgt, 5).tolist()}  sum {wgt.sum():.5f}")

    routed = np.zeros(H)
    for e, wv in zip(eid, wgt):
        routed += wv * expert_raw(ck, L, int(e), x, quant_x=a.quant, quant_act=a.quant)
    sh = expert_raw(ck, L, None, x, quant_x=a.quant, quant_act=a.quant)
    out = routed + sh
    print(f"\n|routed| {np.linalg.norm(routed)/np.sqrt(H):.5f}   |shared| {np.linalg.norm(sh)/np.sqrt(H):.5f}   "
          f"|total| {np.linalg.norm(out)/np.sqrt(H):.5f}   |{label}| {np.linalg.norm(target)/np.sqrt(H):.5f}")
    print(f"vs {label}: relRMS {rr(out, target):.5f}  cos {cos(out, target):+.7f}")


if __name__ == "__main__":
    main()
