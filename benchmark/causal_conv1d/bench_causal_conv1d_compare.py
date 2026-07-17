#!/usr/bin/env python3
# =============================================================================
# causal_conv1d  —  benchmark ONE working-tree build, then COMPARE two runs
# =============================================================================
# Workflow (this is the whole point of the script):
#
#   1. Build the kernel in your working tree (whatever branch is checked out).
#   2. `run` the benchmark  -> saves a results file for THAT build.
#   3. Check out the other branch, rebuild, `run` again -> a second results file.
#   4. `compare` the two results files -> per-shape speedup table + summary.
#
# So you build+run TWICE (once per branch), then compare. Each `run` benchmarks
# whatever .so the working tree currently exposes, auto-detecting the op
# signature, so it works for both the PTO (#572) and AscendC (#592, `main`)
# kernels. EACH shape is timed TWO ways:
#   * e2e     — host-visible latency: warm up, then time each of 100 iterations
#               individually (torch.npu.Event) and take the GEOMETRIC MEAN,
#   * kernel  — pure on-device kernel time: warm up, then profile N=100 iters
#               (torch_npu.profiler) and take the GEOMETRIC MEAN of the per-
#               iteration kernel total (summed Duration in kernel_details.csv).
# plus an fp32-reference correctness check. The sweep is deterministic (fixed
# seeds), so two runs see byte-identical inputs. `compare` reports, per shape,
# the speedup = geomean(baseline) / geomean(other) for each metric.
#
# -----------------------------------------------------------------------------
# PREREQUISITES
#   * Ascend NPU box, CANN sourced:  source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   * Python env with torch + torch_npu.
#   * The working-tree repo BUILT (produces libsgl_kernel_npu.so).
#
# HOW TO RUN
#   export ASCEND_RT_VISIBLE_DEVICES=1        # an IDLE npu (0% AICore, no proc)
#
#   # --- run 1: on `main`, built ---
#   export SGL_KERNEL_SO=/path/to/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so
#   python3 bench_causal_conv1d_compare.py run --label main --out main.json
#
#   # --- run 2: on the branch, rebuilt ---
#   export SGL_KERNEL_SO=/path/to/sgl-kernel-npu-fork/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so
#   python3 bench_causal_conv1d_compare.py run --label tiling --out tiling.json
#
#   # --- compare (first file = baseline) ---
#   python3 bench_causal_conv1d_compare.py compare main.json tiling.json
#
# If SGL_KERNEL_SO is unset, `run` looks for the .so in an importable
# `sgl_kernel_npu` package, then at ./python/.../libsgl_kernel_npu.so.
#
# OPTIONS
#   run:   --label NAME  --out FILE  --layouts 3d,varlen  --csv table.csv
#   env:   QUICK=1                4-shape smoke test
#          SKIP_IDLE_GATE=1       don't abort if the npu looks busy
#          E2E_ITERS=100          e2e iterations (geometric mean) per shape
#          E2E_WARM=10            e2e warmup iterations before timing
#          E2E_ONLY=1             skip the kernel-time profiling (e2e only, faster)
#          KERNEL_N=100           profiler iterations per shape for kernel time
#          KERNEL_WARM=10         kernel warmup iterations before profiling
#          BATCHES=1,16,64  DIMS=2048,4096  SEQS=512   override the sweep grid
#          CAP=<elems>            skip shapes with B*seq*dim above this
#
# `compare` prints BOTH metrics side by side; speedup = baseline / other
#   (>1 => the second build is faster), computed for e2e and for kernel time.
# =============================================================================
import os
import sys
import csv
import glob
import json
import math
import shutil
import argparse
import subprocess

DEV = "npu"
K = 4          # depthwise causal conv width (fp16/bf16)
SL = 4         # conv_state length stored per sequence
DT_NAME = "float16"
TOL = 6e-3
TIE = 0.02     # +/-2% counts as a tie
CAP = int(os.environ.get("CAP", 4_300_000_000))


# ---- sweep grid (overridable via env) ---------------------------------------
def _env_list(name, default):
    v = os.environ.get(name)
    return [int(x) for x in v.split(",")] if v else default


def sweep_plan():
    if os.environ.get("QUICK"):
        return dict(seqs=[512], dims=[4096], batches=[1, 64], layouts=["3d", "varlen"])
    return dict(
        seqs=_env_list("SEQS", [128, 256, 512, 1024]),
        dims=_env_list("DIMS", [1024, 2048, 4096, 5120, 6144]),
        batches=_env_list("BATCHES", [1, 3, 4, 6, 12, 16, 32, 64, 128, 256]),
        layouts=os.environ.get("LAYOUTS", "3d,varlen").split(","),
    )


# ---- environment / device helpers -------------------------------------------
def require_visible_device():
    dev = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not dev:
        sys.exit("ASCEND_RT_VISIBLE_DEVICES is not set. Refusing to default to NPU 0.\n"
                 "Pick an IDLE device explicitly, e.g.  export ASCEND_RT_VISIBLE_DEVICES=1")
    return dev.split(",")[0].strip()


def _aicore_pct(out, dev):
    import re
    f = False
    for ln in out.splitlines():
        if re.search(rf"\|\s*{dev}\s+910", ln):
            f = True
            continue
        if f and ":" in ln:                        # the chip row (has the Bus-Id)
            toks = ln.split()
            for i, t in enumerate(toks):
                if ":" in t:
                    j = i + 1
                    while j < len(toks) and toks[j] == "|":
                        j += 1
                    return toks[j] if j < len(toks) else None
            return None
    return None


def _foreign_procs(out, dev, self_pid):
    """PIDs holding physical npu `dev`, excluding self_pid. Rows are
       `| <npu> <chip> | <pid> | <name> | <mem> |` under the Process id header."""
    procs, in_tbl = [], False
    for ln in out.splitlines():
        if "Process id" in ln:
            in_tbl = True
            continue
        if not in_tbl or f"No running processes found in NPU {dev}" in ln:
            continue
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        if len(parts) >= 3 and parts[0].split() and parts[0].split()[0].isdigit():
            if parts[0].split()[0] == str(dev) and parts[1].isdigit() and parts[1] != str(self_pid):
                procs.append(parts[1])
    return procs


def idle_gate(dev, exclude_self=False):
    """True if physical npu `dev` looks idle (AICore 0% + no foreign process).
       exclude_self drops our own PID (use for the AFTER-run check, when this
       process still holds the NPU context)."""
    if os.environ.get("SKIP_IDLE_GATE"):
        return True
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        print("[gate] npu-smi unavailable — skipping idle check", file=sys.stderr)
        return True
    ai = _aicore_pct(out, dev)
    foreign = _foreign_procs(out, dev, os.getpid() if exclude_self else -1)
    clean = (ai == "0") and not foreign
    print(f"[gate] npu{dev} AICore={ai}% foreign_procs={foreign or 'none'} -> "
          f"{'clean' if clean else 'BUSY'}", file=sys.stderr)
    return clean


def find_so():
    p = os.environ.get("SGL_KERNEL_SO")
    if p:
        if not os.path.exists(p):
            sys.exit(f"SGL_KERNEL_SO={p} does not exist")
        return os.path.abspath(p)
    try:
        import sgl_kernel_npu
        cand = os.path.join(os.path.dirname(sgl_kernel_npu.__file__), "lib", "libsgl_kernel_npu.so")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    rel = "python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
    if os.path.exists(rel):
        return os.path.abspath(rel)
    sys.exit("Could not find libsgl_kernel_npu.so. Build the repo and set "
             "SGL_KERNEL_SO=/path/to/libsgl_kernel_npu.so")


def provenance(so_path):
    """Best-effort md5 + git branch/commit of the repo the .so came from."""
    info = {"so": so_path}
    try:
        info["md5"] = subprocess.run(["md5sum", so_path], capture_output=True,
                                     text=True).stdout.split()[0]
    except Exception:
        info["md5"] = None
    d = os.path.dirname(so_path)
    for _ in range(12):
        if os.path.isdir(os.path.join(d, ".git")):
            g = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True,
                                          text=True).stdout.strip()
            info["git_branch"] = g("rev-parse", "--abbrev-ref", "HEAD")
            info["git_commit"] = g("rev-parse", "--short", "HEAD")
            info["git_dirty"] = bool(g("status", "--porcelain"))
            break
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return info


# ---- kernel-time (torch_npu.profiler) helpers -------------------------------
class _SilenceStdout:
    """Redirect fd 1 to /dev/null so torch_npu.profiler's console output can't
       pollute our tab-separated stdout table. stderr is left untouched."""
    def __enter__(self):
        sys.stdout.flush()
        self._saved = os.dup(1)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self._saved, 1)
        os.close(self._null)
        os.close(self._saved)


def _parse_kernel_csv(pdir, n):
    """Per-iteration on-device time (us) as a GEOMETRIC MEAN over the n profiled
       iterations (matches the e2e statistic). Reconstructs each iteration's
       total kernel time by grouping the per-launch Duration rows (in
       chronological order) by op: an op that fires m launches per call
       contributes its m durations to that iteration. Ops recurring < 0.5*n
       times (one-off setup) are dropped."""
    kd = glob.glob(os.path.join(pdir, "**", "kernel_details.csv"), recursive=True)
    if not kd:
        return None
    with open(kd[0]) as fh:
        r = csv.DictReader(fh)
        if not r.fieldnames:
            return None
        cols = {c.lower().strip(): c for c in r.fieldnames}
        dc = next((cols[c] for c in cols if "duration" in c), None)
        nc = cols.get("name", r.fieldnames[0])
        sc = next((cols[c] for c in cols if "start" in c), None)   # chronological key
        if dc is None:
            return None
        rows = list(r)

    def fnum(row, col):
        try:
            return float(row[col])
        except (TypeError, ValueError):
            return None

    if sc:                                     # sort into launch order
        rows.sort(key=lambda row: (fnum(row, sc) if fnum(row, sc) is not None else 0.0))
    order = {}
    for row in rows:
        v = fnum(row, dc)
        if v is None:
            continue
        nm = (row.get(nc) or "").strip()
        order.setdefault(nm, []).append(v)
    qual = {nm: ds for nm, ds in order.items() if len(ds) >= 0.5 * n}
    if not qual:
        return None
    per_iter = [0.0] * n
    for nm, ds in qual.items():
        m = max(1, round(len(ds) / n))         # launches of this op per iteration
        for i in range(n):
            per_iter[i] += sum(ds[i * m:(i + 1) * m])
    vals = [t for t in per_iter if t > 0]
    if not vals:
        return None
    return round(math.exp(sum(math.log(t) for t in vals) / len(vals)), 2)


# =============================================================================
# RUN  — benchmark the current working-tree build, save a results file
# =============================================================================
def do_run(args):
    import torch
    import torch.nn.functional as F
    import torch_npu  # noqa: F401
    import itertools

    dev = require_visible_device()
    so = find_so()
    prov = provenance(so)
    if not idle_gate(dev):
        sys.exit(f"npu{dev} is not idle — aborting (set SKIP_IDLE_GATE=1 to override)")

    DT = getattr(torch, DT_NAME)
    torch.ops.load_library(so)
    torch.npu.set_device(DEV)
    torch.manual_seed(0)

    def _call_pto(x, w, b, cs, qsl, ci, hi):
        return torch.ops.npu.causal_conv1d(x, w, cs, qsl, ci, hi,
                                           bias=b, activation_mode=True, pad_slot_id=-1)

    def _call_asc(x, w, b, cs, qsl, ci, hi):
        return torch.ops.npu.causal_conv1d(x, w, cs, b, qsl, ci, hi, None, 1, -1, 0)

    def detect():
        B, L, d = 2, 8, 16
        z = lambda *s: torch.zeros(s, device=DEV, dtype=DT)
        x, w, b, cs = z(B, L, d), z(K, d), z(d), z(B, SL, d)
        qsl = torch.arange(0, (B + 1) * L, L, device=DEV, dtype=torch.int32)
        ci = torch.arange(0, B, device=DEV, dtype=torch.int32)
        hi = torch.zeros((B,), device=DEV, dtype=torch.bool)
        for fn, name in ((_call_pto, "PTO #572"), (_call_asc, "ASC #592")):
            try:
                fn(x, w, b, cs, qsl, ci, hi); torch.npu.synchronize()
                return fn, name
            except Exception:
                pass
        sys.exit("neither known causal_conv1d signature works for this .so")

    call, variant = detect()

    E2E_ITERS = int(os.environ.get("E2E_ITERS", 100))
    E2E_WARM = int(os.environ.get("E2E_WARM", 10))

    def ev(fn, iters=E2E_ITERS, warm=E2E_WARM):
        # warm up, then time EACH iteration individually and return the
        # geometric mean of the per-call latencies (us).
        for _ in range(warm):
            fn()
        torch.npu.synchronize()
        acc = 0.0
        for _ in range(iters):
            s = torch.npu.Event(True); e = torch.npu.Event(True)
            s.record(); fn(); e.record()
            torch.npu.synchronize()
            t = s.elapsed_time(e) * 1e3                 # ms -> us, single call
            acc += math.log(t if t > 1e-6 else 1e-6)
        return math.exp(acc / iters)                    # geometric mean

    KERNEL_N = int(os.environ.get("KERNEL_N", 100))
    KERNEL_WARM = int(os.environ.get("KERNEL_WARM", 10))
    E2E_ONLY = bool(os.environ.get("E2E_ONLY"))
    kprof_dir = f"/tmp/cc_kprof_{dev}_{os.getpid()}"

    def kernel_us(fn, n=KERNEL_N, warm=KERNEL_WARM):
        for _ in range(warm):                 # warm up THIS shape before profiling
            fn()
        torch.npu.synchronize()
        for _ in range(4):                    # retry: the first profile can miss the CSV / TASK table
            shutil.rmtree(kprof_dir, ignore_errors=True)
            os.makedirs(kprof_dir, exist_ok=True)
            with _SilenceStdout():
                with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU],
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(kprof_dir)):
                    for _ in range(n):
                        fn()
                    torch.npu.synchronize()
            us = _parse_kernel_csv(kprof_dir, n)
            if us is not None:
                return us
        return None

    def make_3d(B, dim, L):
        x = 2 * torch.rand((B, L, dim), device=DEV, dtype=DT) - 1
        w = (torch.rand((K, dim), device=DEV, dtype=DT) - 0.5).contiguous()
        b = (torch.rand((dim,), device=DEV, dtype=DT) - 0.5).contiguous()
        cs = torch.zeros((B, SL, dim), device=DEV, dtype=DT)
        qsl = torch.arange(0, (B + 1) * L, L, device=DEV, dtype=torch.int32)
        ci = torch.arange(0, B, device=DEV, dtype=torch.int32)
        hi = torch.zeros((B,), device=DEV, dtype=torch.bool)
        nb = min(B, max(1, 200_000_000 // (L * dim)))
        pad = torch.zeros((nb, K - 1, dim), device=DEV, dtype=DT)
        xe = torch.cat([pad, x[:nb]], 1).float()
        yr = F.silu(sum(xe[:, k:k + L] * w.float()[k] for k in range(K)) + b.float())
        return (x, w, b, cs, qsl, ci, hi), yr, (B, L, dim), nb

    def make_varlen(B, dim, L):
        g = torch.Generator().manual_seed(B * 1000 + dim)
        lens = torch.randint(L // 2, L + L // 2 + 1, (B,), generator=g).tolist()
        total = sum(lens)
        qsl = torch.tensor([0] + list(itertools.accumulate(lens)), device=DEV, dtype=torch.int32)
        x = 2 * torch.rand((total, dim), device=DEV, dtype=DT) - 1
        w = (torch.rand((K, dim), device=DEV, dtype=DT) - 0.5).contiguous()
        b = (torch.rand((dim,), device=DEV, dtype=DT) - 0.5).contiguous()
        cs = torch.zeros((B, SL, dim), device=DEV, dtype=DT)
        ci = torch.arange(B, device=DEV, dtype=torch.int32)
        hi = torch.zeros((B,), device=DEV, dtype=torch.bool)
        cap = max(1, 200_000_000 // dim)
        nbtok = nbs = 0
        for Lg in lens:
            if nbtok + Lg > cap and nbs > 0:
                break
            nbtok += Lg; nbs += 1
        outs, off = [], 0
        for Lg in lens[:nbs]:
            xi = x[off:off + Lg]
            pad = torch.zeros((K - 1, dim), device=DEV, dtype=DT)
            xe = torch.cat([pad, xi], 0).float()
            outs.append(F.silu(sum(xe[k:k + Lg] * w.float()[k] for k in range(K)) + b.float()))
            off += Lg
        return (x, w, b, cs, qsl, ci, hi), torch.cat(outs, 0), (total, dim), nbtok

    plan = sweep_plan()
    layouts = [x for x in (args.layouts.split(",") if args.layouts else plan["layouts"]) if x]
    print(f"# label={args.label}  variant={variant}  git={prov.get('git_branch')}@{prov.get('git_commit')}"
          f"{'(dirty)' if prov.get('git_dirty') else ''}  md5={prov.get('md5')}", file=sys.stderr)
    method = f"e2e(geomean,{E2E_WARM}warm+{E2E_ITERS}iters)" + \
             ("" if E2E_ONLY else f"+kernel(prof,{KERNEL_WARM}warm+geomean/N={KERNEL_N})")
    print(f"# device=npu{dev}  method={method}  dtype={DT_NAME}  tol={TOL}", file=sys.stderr)

    if not E2E_ONLY:                          # prime the profiler (its first call often yields no table)
        try:
            ptup, _, _, _ = make_3d(2, 16, 32)
            kernel_us(lambda: call(*ptup), n=5)
            del ptup
            torch.npu.empty_cache()
        except Exception:
            pass

    records, n_ok, n_wrong, n_err = [], 0, 0, 0
    hdr = ["layout", "B", "dim", "seq", "B*seq*dim", "e2e_us", "kernel_us", "ok"]
    print("\t".join(hdr))
    for layout in layouts:
        mk = make_3d if layout == "3d" else make_varlen
        for L in plan["seqs"]:
            for dim in plan["dims"]:
                for B in plan["batches"]:
                    if B * L * dim > CAP:
                        continue
                    rec = {"layout": layout, "B": B, "seq": L, "dim": dim, "elems": B * L * dim}
                    try:
                        tup, yr, shp, nb = mk(B, dim, L)
                        y = call(*tup); torch.npu.synchronize()
                        err = (y.reshape(*shp)[:nb].float() - yr).abs().max().item()
                        rec["ok"] = "ok" if err <= TOL else "WRONG"
                        n_ok += rec["ok"] == "ok"; n_wrong += rec["ok"] == "WRONG"
                        del yr
                        rec["e2e_us"] = round(ev(lambda: call(*tup)), 2)
                        rec["kernel_us"] = None if E2E_ONLY else kernel_us(lambda: call(*tup))
                        del tup, y
                    except RuntimeError as ex:
                        rec["ok"] = "ERR/" + type(ex).__name__
                        rec["e2e_us"] = rec["kernel_us"] = None
                        n_err += 1
                    records.append(rec)
                    print("\t".join((str(rec["elems"]) if c == "B*seq*dim"
                                     else ("nan" if rec.get(c) is None else str(rec[c])))
                                    for c in hdr))
                    torch.npu.empty_cache()

    if not idle_gate(dev, exclude_self=True):
        print(f"# WARNING: npu{dev} not clean AFTER the run — a foreign job appeared mid-run",
              file=sys.stderr)
    out = args.out or f"cc_bench_{args.label}.json"
    with open(out, "w") as fh:
        json.dump({"label": args.label, "variant": variant, "device": dev,
                   "kernel_n": None if E2E_ONLY else KERNEL_N,
                   "provenance": prov, "plan": plan, "records": records}, fh, indent=1)
    print(f"# done: {n_ok} ok, {n_wrong} WRONG, {n_err} err  ->  {out}", file=sys.stderr)


# =============================================================================
# COMPARE  — join two results files, print per-shape speedup + summary
# =============================================================================
def _load(path):
    with open(path) as fh:
        d = json.load(fh)
    idx = {(r["layout"], r["B"], r["seq"], r["dim"]): r for r in d["records"]}
    return d, idx


def do_compare(args):
    A, Ai = _load(args.baseline)
    Bd, Bi = _load(args.other)
    la, lb = A["label"], Bd["label"]
    ap = A.get("provenance", {}); bp = Bd.get("provenance", {})
    print(f"# baseline {la}: {A.get('variant')}  {ap.get('git_branch')}@{ap.get('git_commit')}  md5={ap.get('md5')}",
          file=sys.stderr)
    print(f"# other    {lb}: {Bd.get('variant')}  {bp.get('git_branch')}@{bp.get('git_commit')}  md5={bp.get('md5')}",
          file=sys.stderr)
    print(f"# columns: *_e2e = host latency (us), *_kernel = on-device kernel time (us); "
          f"speedup_e2e/speedup_kernel = {la}/{lb}  (>1 => {lb} faster)", file=sys.stderr)

    def e2e_of(r): return r.get("e2e_us", r.get("us"))     # back-compat with old "us" files
    def ker_of(r): return r.get("kernel_us")

    keys = sorted(set(Ai) & set(Bi), key=lambda k: (k[0], k[3], k[1]))
    hdr = ["layout", "B", "dim", "seq", "B*seq*dim",
           f"{la}_e2e", f"{lb}_e2e", "speedup_e2e",
           f"{la}_kernel", f"{lb}_kernel", "speedup_kernel", "ok"]
    print("\t".join(hdr))
    r_e2e, r_ker = {}, {}
    win_e2e = {la: 0, lb: 0, "tie": 0}
    win_ker = {la: 0, lb: 0, "tie": 0}
    wrong, rows = 0, []
    for k in keys:
        ra, rb = Ai[k], Bi[k]
        oka, okb = ra.get("ok"), rb.get("ok")
        ok = "ok" if oka == "ok" and okb == "ok" else f"{la}:{oka}/{lb}:{okb}"
        if "WRONG" in (oka, okb):
            wrong += 1

        def ratio(av, bv, store, win):
            if av and bv:
                sp = av / bv
                store.setdefault(k[0], []).append(sp); store.setdefault("all", []).append(sp)
                win[lb if sp > 1 + TIE else (la if sp < 1 - TIE else "tie")] += 1
                return f"{sp:.3f}"
            return "n/a"

        ae, be = e2e_of(ra), e2e_of(rb)
        ak, bk = ker_of(ra), ker_of(rb)
        xe = ratio(ae, be, r_e2e, win_e2e)
        xk = ratio(ak, bk, r_ker, win_ker)
        fmt = lambda v: "nan" if v is None else f"{v:.1f}"
        row = [k[0], k[1], k[3], k[2], ra["elems"],
               fmt(ae), fmt(be), xe, fmt(ak), fmt(bk), xk, ok]
        print("\t".join(str(c) for c in row)); rows.append(row)

    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")

    def med(xs):
        s = sorted(xs); n = len(s)
        return float("nan") if not n else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

    def block(name, store, win):
        if not store.get("all"):
            print(f"# {name}: (no data — one side lacks this metric)", file=sys.stderr); return
        print(f"# {name}  (speedup = {la}/{lb}, >1 => {lb} faster)", file=sys.stderr)
        for lay in ["3d", "varlen", "all"]:
            xs = store.get(lay, [])
            if xs:
                print(f"#   {lay:6s} n={len(xs):3d}  median={med(xs):.3f}  geomean={gm(xs):.3f}  "
                      f"min={min(xs):.3f}  max={max(xs):.3f}", file=sys.stderr)
        print(f"#   wins: {lb}={win[lb]}  {la}={win[la]}  tie={win['tie']}", file=sys.stderr)

    print("\n# ===== summary =====", file=sys.stderr)
    block("E2E   ", r_e2e, win_e2e)
    block("KERNEL", r_ker, win_ker)
    print(f"# WRONG shapes={wrong}", file=sys.stderr)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
        print(f"# wrote {args.csv}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Benchmark a working-tree build, or compare two runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark the current working-tree build")
    r.add_argument("--label", default="build", help="name for this build (e.g. main / tiling)")
    r.add_argument("--out", default=None, help="results file to write (default cc_bench_<label>.json)")
    r.add_argument("--layouts", default=None, help="3d,varlen (default: both)")
    r.set_defaults(func=do_run)

    c = sub.add_parser("compare", help="compare two results files (first = baseline)")
    c.add_argument("baseline", help="baseline results file (e.g. main.json)")
    c.add_argument("other", help="other results file (e.g. tiling.json)")
    c.add_argument("--csv", default=None, help="also write the joined table to this CSV")
    c.set_defaults(func=do_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
