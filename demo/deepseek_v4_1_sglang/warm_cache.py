"""Fill the static-kernel build cache from an existing build tree, in parallel.

`compile_static` rebuilds a layer whenever `MIRAGE_STATIC_BUILD_CACHE` is unset -- and `full_decode.py` used to leave it
unset, so all 80 layers were nvcc'd on every run even though `<out>/a{L}/build/layer.cu` was byte-identical to what the
run before had compiled. With the cache set, the run skips a layer whose key is already there; this script puts the keys
there, so the first run after a change of `--out`/`--dump-dir` is not a rebuild either.

The key is exactly `compile_static`'s: md5(layer.cu) + the flags + `code_tag()` (the kernel templates' md5). Only the
arch is taken from this host, so the entries match the .so the same host built.

  python warm_cache.py --out /tmp/full_real              # register what that tree already compiled (no nvcc)
  python warm_cache.py --out /tmp/full_real --jobs 16    # ... and nvcc, in parallel, the layers it is missing
  python warm_cache.py --out /tmp/full_real --dry-run    # what the keys are and which are missing

`--out` is `full_decode.py --out`: it holds `a{L}/build/layer.cu` and `m{L}/build/layer.cu`.
"""
import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from mirage.mpk.static_schedule import code_tag   # noqa: E402  (the templates' md5: part of the key)


def arch_of(gpu=0):
    """compile_command's arch string for this host's GPUs (sm_{arch})."""
    import torch
    major, minor = torch.cuda.get_device_capability(gpu)
    return f"{major}{minor}a" if major >= 9 else f"{major}{minor}"


def key_of(cu_path, arch, extra_flags):
    """compile_static's md5: the source, the flags it passes nvcc (command[1:5]), then code_tag()."""
    command_tail = ["-O3", "-std=c++17", "-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    h = hashlib.md5()
    with open(cu_path, "rb") as f:
        h.update(f.read())
    h.update(" ".join(command_tail + list(extra_flags or [])).encode())
    h.update(code_tag().encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="a full_decode --out tree (a{L}/build, m{L}/build)")
    ap.add_argument("--jobs", type=int, default=8, help="nvcc processes for the layers the cache is missing (0: none)")
    ap.add_argument("--arch", default=None, help="sm arch (default: this host's GPU)")
    ap.add_argument("--flag", action="append", default=[], help="an extra nvcc flag, as full_decode --flag")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cache = os.environ.get("MIRAGE_STATIC_BUILD_CACHE", "/tmp/mirage_static_cache")
    os.makedirs(cache, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    arch = a.arch or arch_of(0)

    cus = sorted(glob.glob(os.path.join(a.out, "*", "build", "layer.cu")))
    if not cus:
        sys.exit(f"no layer.cu under {a.out}/*/build (is --out a full_decode build tree?)")
    have, missing, copied = 0, [], 0
    for cu in cus:
        k = key_of(cu, arch, a.flag)
        dst = os.path.join(cache, k + suffix)
        if os.path.exists(dst):
            have += 1
            continue
        so = glob.glob(os.path.join(os.path.dirname(cu), "layer*" + suffix))
        if so and not a.dry_run:
            # this tree already compiled it: register the .so under the key instead of nvcc'ing it again
            tmp = dst + ".tmp"
            shutil.copy(so[0], tmp)
            os.replace(tmp, dst)
            copied += 1
        else:
            missing.append((os.path.dirname(cu), cu, dst))

    print(f"arch sm_{arch}  code_tag {code_tag()}  cache {cache}")
    print(f"{len(cus)} layers: {have} already cached, {copied} registered from this tree, {len(missing)} to compile")
    if a.dry_run or not missing or a.jobs <= 0:
        return

    def build(item):
        d, cu, dst = item
        cmd = [shutil.which("nvcc"), "-O3", "-std=c++17", "-gencode",
               f"arch=compute_{arch},code=sm_{arch}", "-shared", "-Xcompiler=-fPIC", "-lcuda", cu,
               "-o", os.path.join(d, "warm" + suffix)] + list(a.flag)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAILED {cu}: {r.stderr[-400:]}"
        tmp = dst + ".tmp"
        shutil.copy(os.path.join(d, "warm" + suffix), tmp)
        os.replace(tmp, dst)
        return f"built {os.path.basename(os.path.dirname(d))}"

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for i, msg in enumerate(ex.map(build, missing), 1):
            print(f"[{i}/{len(missing)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
