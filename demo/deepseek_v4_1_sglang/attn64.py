"""One swa attention sublayer of DeepSeek-V4.1 recomputed in float64, to say WHOSE error the last 0.8% is.

The chain's L1 attention matches SGLang to 0.78% (teacher-forced: exact input, exact coefficients). That number only means
something once we know how far a computation that is *right but rounded differently* lands from SGLang -- SGLang's own
bf16/fp8 path is one sample, not the truth. So this tool builds the sublayer in fp64 from the checkpoint's own weights
(dequantized exactly: an e4m3/e2m1 value times 2^k has at most 4 significant bits) and SGLang's own K pool, then prints
three y's side by side:

    y_fp64    the float64 reference, rounded to bf16 only at SGLang's own rounding points
    y_gold    SGLang's, read back out of its dumps through  R' = post y + comb^T R
    y_chain   the static chain's, likewise

Whoever is closer to fp64 carries less error. All three get the SAME input residual (golden) and the SAME coefficients.

  python attn64.py --layer 1                 # the number that matters: |y_chain - y_fp64| vs |y_gold - y_fp64|
  python attn64.py --layer 0                 # a control: SGLang and fp64 must agree here (L0 is reproducible bit-exactly)
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

HID, D, RD, HEADS, GROUPS, OL = 5120, 512, 64, 64, 8, 1024
QL = 1280
W = 128
EPS = 1e-20
PRE, CK = "/tmp/dec_coef8", "/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017"
IDX = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]


def read(n):
    with safe_open(os.path.join(CK, IDX[n]), "pt") as sf:
        return sf.get_tensor(n)


def deq(name, rows=None):
    """e4m3 weight + 32x32 e8m0 block scales -> fp64, exactly (4 significant bits). `rows`: a 32-block-aligned slice."""
    w = read(name + ".weight")
    s = read(name + ".scale").float()
    if rows is not None:
        r0, r1 = rows
        w = w[r0:r1]
        s = s[r0 // 32:r1 // 32]
    R, C = w.shape
    return (w.float().view(R // 32, 32, C // 32, 32) * s[:, None, :, None]).reshape(R, C).double().numpy()


def rms(x, g, eps=EPS):
    x = np.asarray(x, np.float64)
    return x / np.sqrt((x ** 2).mean() + eps) * g


def rope(x, pos, theta=10000.0, inverse=False):
    """Plain RoPE on the last RD entries, adjacent pairs as complex numbers (model.py apply_rotary_emb)."""
    x = np.array(x, np.float64)
    z = x[..., -RD:].reshape(-1, RD // 2, 2)
    inv = 1.0 / theta ** (np.arange(0, RD, 2) / RD)
    ang = (-1 if inverse else 1) * pos * inv
    c, s = np.cos(ang), np.sin(ang)
    re, im = z[..., 0].copy(), z[..., 1].copy()
    z[..., 0] = re * c - im * s
    z[..., 1] = re * s + im * c
    x[..., -RD:] = z.reshape(-1, RD)
    return x


def bf16(x):
    return torch.from_numpy(np.asarray(x)).float().bfloat16().float().numpy()


def act_quant(x, block=32):
    """kernel.py act_quant(inplace=True): e4m3, `block`-wide groups, power-of-two (e8m0) scale -> bf16."""
    v = torch.from_numpy(np.asarray(x, np.float32)).float().view(-1, block)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    ex = torch.ceil(torch.log2(amax / 448.0))
    q = (v / torch.pow(2.0, ex)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * torch.pow(2.0, ex)
    return q.reshape(-1).bfloat16().float().numpy()


def pool_row(L, pos):
    """The window K rows of layer L in the official's list order (oldest first), from SGLang's own pool.

    A row is 528 B: 512 e4m3 + 16 e8m0 scales, one token's entry in a 256-token page. `None` = the current token (written
    by this launch, so absent from the dump); the caller supplies it."""
    meta = json.load(open(os.path.join(PRE, "meta_rank00.json")))["buffers"]
    addr = json.load(open(os.path.join(PRE, "meta_addr_rank00.json")))
    per = 135168 // 528
    p = np.fromfile(os.path.join(PRE, meta[f"L{L}.window_cache"]["file"]), np.uint8).reshape(-1, per, 528)
    loc = {int(a): int(b) for a, b in zip(addr["positions"], addr["swa_out_cache_loc"])}
    out = []
    for s in [(pos % W + 1 + i) % W for i in range(W)]:
        q = pos - ((pos - s) % W)
        if q == pos:
            out.append(None)
            continue
        c = loc[q]
        b = p[c // per, c % per]
        v = torch.frombuffer(bytearray(b[:512].tobytes()), dtype=torch.float8_e4m3fn).float().numpy()
        sc = torch.frombuffer(bytearray(b[512:].tobytes()), dtype=torch.uint8).int().numpy() - 127
        out.append((v.reshape(16, 32) * np.pow(2.0, sc)[:, None]).reshape(512))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pos", type=int, default=31906)
    ap.add_argument("--chain", default="/tmp/chain40")
    a = ap.parse_args()
    L, pos = a.layer, a.pos
    p = f"layers.{L}.attn."
    G = os.path.join(PRE, "layers", "step1")
    cf = np.fromfile(os.path.join(a.chain, f"attn{L}_coef.bin"), np.float32).reshape(-1, 32)[0]
    post, comb = cf[4:8].astype(np.float64), cf[8:24].reshape(4, 4).astype(np.float64)

    # ---- the residual this sublayer reads, and the two outputs, all through R' = post y + comb[i][j] R_i
    def resid(name):
        return np.fromfile(os.path.join(G, name + ".f32"), np.float32).reshape(4, HID).astype(np.float64)

    def streams(name):
        t = torch.frombuffer(bytearray(open(os.path.join(a.chain, name + "_streams_out.bin"), "rb").read()),
                             dtype=torch.bfloat16).float()
        return t.reshape(-1, 4, HID)[0].numpy().astype(np.float64)

    # the residual this sublayer reads: L00_in at 0; at L1 the Engram host adds eg_emb to L00_apost BEFORE the block
    # (model.py: `if layer.engram is not None: h = layer.engram(...)` at the top of the loop), so its input is L01_eg_y
    if L == 0:
        rin_g = resid("L00_in")
    elif os.path.exists(os.path.join(G, f"L{L:02d}_eg_y.f32")):
        rin_g = resid(f"L{L:02d}_eg_y")
    else:
        rin_g = resid(f"L{L - 1:02d}_mpost")
    rout_g = resid(f"L{L:02d}_apost")
    rout_c = streams(f"attn{L}")
    rin_c = rin_g if L == 0 else streams(f"moe{L - 1}")   # L0's input is the embedding, which is not a sublayer's dump

    # ---- the sublayer itself, in fp64. `x` = hc_pre(attn L) + attn_norm: the identity mix at L0, else moe{L-1}'s ffn_pre
    # (see FINDINGS §12.1: model.py hands the mix down from the previous sublayer)
    if L == 0:
        pre_mix = np.array([1.0, 0, 0, 0])
    else:
        c = np.fromfile(os.path.join(a.chain, f"moe{L - 1}_coef.bin"), np.float32).reshape(-1, 32)[0]
        pre_mix = c[:4].astype(np.float64)
    x = rms(pre_mix @ rin_g, read(f"layers.{L}.attn_norm.weight").double().numpy())

    qr = rms(deq(p + "wq_a") @ x, read(p + "q_norm.weight").double().numpy())
    q = rope((deq(p + "wq_b") @ qr).reshape(HEADS, D), pos)          # [64, 512], fp64

    k_now = rms(deq(p + "wkv") @ x, read(p + "kv_norm.weight").double().numpy())
    k_now = act_quant(rope(k_now, pos), 32)                          # what the kernel writes into the pool
    rows = pool_row(L, pos)
    assert rows[-1] is None, "the current token is the newest list position"
    kv = np.stack([k_now if r is None else r for r in rows])         # [128, 512]
    sink = read(p + "attn_sink").double().numpy()

    s = (q @ kv.T) / np.sqrt(D)
    m = s.max(1)
    e = np.exp(s - m[:, None])
    o = (e @ kv) / (e.sum(1) + np.exp(sink - m))[:, None]
    o = rope(o, pos, inverse=True)

    # o -> y: the server feeds bf16 `o` straight into a bf16 wo_a (no act_quant -- see FINDINGS 12.9: the chain's own
    # wo_a_part buffer reproduces this to 2.9e-7, while act_quant(o) is off by 3.5%), then wo_b over the block-diagonal
    # 8x1024 layout. Each group sees 8 heads.
    aq = o.reshape(GROUPS, HEADS // GROUPS * D)
    wa = np.concatenate([deq(p + "wo_a", (g * OL, (g + 1) * OL)) for g in range(GROUPS)])   # [8192, 4096]
    bb = np.stack([wa[g * OL:(g + 1) * OL] @ aq[g] for g in range(GROUPS)])                 # [8, 1024]
    y64 = deq(p + "wo_b") @ bb.reshape(-1)                                                  # [5120]

    # SGLang's wo_a is bf16 (convert.py dequantizes it), so o*fp8 then a bf16 wo_a... the fp64 reference keeps wo_a exact:
    # the dequantized value IS bf16-representable, so this adds no error of its own.
    print(f"L{L} attention: a float64 reference (golden input, golden K pool, exact weights) vs the two implementations")
    print("  forward comparison through R' = post*y + comb[i][j]*R_i, bf16-rounded as SGLang's _mhc_post_torch does:")
    for tag, rout in (("SGLang", rout_g), ("static chain", rout_c)):
        rp = bf16(comb.T @ rin_g + post[:, None] * y64[None, :])
        r = float(np.linalg.norm(rp - rout) / max(np.linalg.norm(rout), 1e-30))
        c = float((rp.ravel() @ rout.ravel()) / max(np.linalg.norm(rp) * np.linalg.norm(rout), 1e-30))
        print(f"    R' from fp64 vs {tag:12s} relRMS {r:.5f}  cos {c:+.7f}")
    r = float(np.linalg.norm(rout_c - rout_g) / max(np.linalg.norm(rout_g), 1e-30))
    print(f"    the two dumps against each other     relRMS {r:.5f}")
    print(f"  |y_fp64| rms {np.linalg.norm(y64) / np.sqrt(HID):.5f}   post {np.round(post, 4)}")


if __name__ == "__main__":
    main()
