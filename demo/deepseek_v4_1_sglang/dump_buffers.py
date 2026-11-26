"""Which device buffer actually held the bad bytes, read back from the pointers the kernel printed.

The question this answers: when the pure static chain goes NaN, is the NaN in a buffer the layer resets each
launch (a local sentinel read early) or in the exchange region (a peer that never published this launch)? The
dumps in e2e_decode cover the tensors the layer declares and the exchange pool, but not the scratch buffers
inside the kernel -- and those are exactly the 0xFF-filled, polled ones.

DSV41_RESET_DEBUG=1 makes the layer print, before the run, one line per reset entry (core.cuh / host.cuh):
  [rl] rank 0: p 0x7f... bytes 65536 stride 0 v 255 by_reader 0
That is every buffer the kernel fills at the start of a launch, or the reader re-arms -- with its address.
This reads each one back after the step and says how much of it is NaN (as float32) and 0xFF.

  # run with the layer's stderr captured, then:
  python dump_buffers.py --log /tmp/e2e_dbg.log --rank 0 --layer-cu /tmp/.../attn4_t1/build/layer.cu

Names: the [rl] lines carry none, so they are matched against the layer's own declarations, which layer.cu
spells out as {"buf.<slot>", "<kind> <name> <bytes> <reset> <by reader>"}, on (bytes, v, by_reader). A match
is reported with the name; a miss is reported by its address rather than guessed.

Safety: cudaMemcpy from a raw address segfaults if the address is stale (the layer was freed, or the log is
from another process). Pass --rank for the same GPU the run used, and only point this at a log whose process
is still alive (e2e_decode with DSV41_E2E_ONE=1 and a sleep, or the same script doing the read back itself).
"""
import argparse
import re
import sys

DECL = re.compile(r'\{"buf\.(\d+)",\s*"(\w+)\s+(\S+)\s+(\d+)\s+(-?\d+)\s+(\d+)"\}')
RESET = re.compile(
    r"\[rl\]\s+rank\s+(\d+):\s+p\s+(0x[0-9a-fA-F]+)\s+bytes\s+(\d+)\s+stride\s+(\d+)\s+"
    r"v\s+(\d+)\s+by_reader\s+(\d+)")


def declarations(layer_cu):
    """[(kind, name, bytes, reset, by_reader)] in slot order, from the layer's own init map."""
    out = []
    for line in open(layer_cu, errors="replace"):
        for m in DECL.finditer(line):
            out.append((m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(6))))
    return out


def resets(log):
    out = []
    for line in open(log, errors="replace"):
        for m in RESET.finditer(line):
            out.append(dict(rank=int(m.group(1)), ptr=int(m.group(2), 16), bytes=int(m.group(3)),
                            stride=int(m.group(4)), v=int(m.group(5)), by_reader=int(m.group(6))))
    return out


def analyze(rank, entries, decls, copy=None):
    """[(label, bytes, v, by_reader, n_nan, n_ff, n_zero, first_nan)] for each entry, read back from the device.

    copy: a callable (dev_ptr, n) -> bytes, so a caller inside the run (e2e_decode) can pass its own reader.
    The default one uses cuda.bindings on the current process's context, which is only right for the process
    that allocated the pointers -- never for another one: a raw address from a log belongs to that process.
    """
    if copy is None:
        from cuda.bindings import runtime as rt

        def copy(dev_ptr, n):
            import ctypes
            buf = ctypes.create_string_buffer(n)
            (err,) = rt.cudaMemcpy(buf, dev_ptr, n, rt.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            if err != rt.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpy({dev_ptr:#x}, {n}): {err}")
            return buf.raw[:n]

    live = [d for d in decls if d[3] >= 0]   # a declaration with reset -1 is not a reset entry
    used, rows, seen = set(), [], set()
    for e in entries:
        if e["ptr"] in seen:
            continue
        seen.add(e["ptr"])
        raw = copy(e["ptr"], e["bytes"])
        nff = raw.count(0xFF)
        nz = sum(1 for b in raw if b)
        n_nan, first = 0, None
        for off in range(0, len(raw) - (len(raw) % 4), 4):
            w = int.from_bytes(raw[off:off + 4], "little")
            if (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF):
                n_nan += 1
                if first is None:
                    first = off // 4
        nm = None
        if live:
            # 1:1 by order: the reset list is built in declaration order (host.cuh walks s.resets), so among
            # equal-sized entries with the same fill and side, the first unused declaration is this one's.
            def take(strict):
                for i, d in enumerate(live):
                    if i in used:
                        continue
                    if d[2] == e["bytes"] and d[3] == e["v"] and (not strict or d[4] == e["by_reader"]):
                        used.add(i)
                        return d[1]
                return None
            nm = take(True) or take(False)
        label = nm or f"{e['ptr']:#x}"
        if nm is None and e["v"] == 0 and e["bytes"] % 4 == 0 and e["bytes"] <= 4096:
            label = f"{e['ptr']:#x} (a counter)"
        rows.append((label, e["bytes"], e["v"], e["by_reader"], n_nan, nff, nz, first))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="the run's stderr, with DSV41_RESET_DEBUG=1")
    ap.add_argument("--rank", type=int, default=0, help="the GPU to read (the log's rank)")
    ap.add_argument("--layer-cu", default=None, help="the layer's generated layer.cu, to name the buffers")
    ap.add_argument("--max", type=int, default=96, help="at most this many entries (the counters are many)")
    a = ap.parse_args()

    from cuda.bindings import runtime as rt

    (err,) = rt.cudaSetDevice(a.rank)
    if err != rt.cudaError_t.cudaSuccess:
        raise SystemExit(f"cudaSetDevice({a.rank}): {err}")

    ents = [e for e in resets(a.log) if e["rank"] == a.rank]
    if not ents:
        raise SystemExit(f"no '[rl] rank {a.rank}' line in {a.log}: rerun with DSV41_RESET_DEBUG=1")
    decls = declarations(a.layer_cu) if a.layer_cu else []

    print(f"rank {a.rank}: {len(ents)} reset entries"
          + (f", {len(decls)} declared buffers in {a.layer_cu}" if decls else ""))
    print(f"{'buffer':<26} {'bytes':>9} {'v':>3} {'rd':>2} {'NaN':>8} {'0xFF':>9} {'nonzero':>9}  first NaN")
    for label, n, v, rd, n_nan, nff, nz, first in analyze(a.rank, ents[:a.max], decls):
        print(f"{label:<26} {n:>9} {v:>3} {rd:>2} {n_nan:>8} {nff:>9} {nz:>9}  {first}")
        print(f"{label:<26} {n:>9} {e['v']:>3} {e['by_reader']:>2} {n_nan:>8} {nff:>9} {nz:>9}  {first}")
    print("\nread it as: NaN>0 in a v=255 buffer the layer resets each launch -> a local sentinel was consumed\n"
          "            before its producer wrote it; NaN in an exchange buffer -> a peer published into the\n"
          "            other set this launch (FINDINGS_correctness.md 4.3). nonzero==0 with v=255 says the\n"
          "            layer's own fill is gone, i.e. the producer wrote and the reader re-armed it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
