# causal_conv1d — compare `main` vs a branch (e2e **and** kernel time)

Build a kernel, `run` it, do the same for the other branch, then `compare`.
Each `run` measures **both** metrics per shape (after warmup):
- **e2e** — host latency: 100 iterations timed individually, reported as their **geometric mean**.
- **kernel** — on-device kernel time: 100 profiled iterations, reported as the **geometric mean** of the per-iteration kernel total.

### 0. Set these once (edit for your setup, then paste in your shell)
Keep the **same terminal** for all steps below — these variables live in that shell.
```bash
# the two repos to compare (edit to your clone locations):
MAIN_REPO=/path/to/sgl-kernel-npu          # baseline repo
BRANCH_REPO=/path/to/sgl-kernel-npu-fork   # branch repo

# branch to check out in each repo:
MAIN_BRANCH=main
FEAT_BRANCH=causal-conv1d-tiling

# directory holding bench_causal_conv1d_compare.py and the result files:
TOOLS=/path/to/benchmark-scripts

# built-kernel path inside a repo (standard — leave as is):
SO_REL=python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so
```

### 1. Environment

**One-time — create a venv with torch + torch_npu** (skip if you already have one):
```bash
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install --upgrade pip setuptools wheel
# torch_npu MUST match your CANN version (2.10.0 <-> CANN 8.x; tested on CANN 8.5.0)
pip install torch==2.10.0 torch-npu==2.10.0
pip install numpy==1.26.4 pyyaml psutil decorator attrs scipy pybind11
```
Reference env: Python 3.12, torch 2.10.0, torch_npu 2.10.0, CANN 8.5.0.
(`pyyaml` is needed at `import torch_npu` time; `psutil` by the kernel build.)

**Every new terminal — source CANN + activate the venv:**
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # your CANN toolkit set_env.sh
source /path/to/venv/bin/activate
```

### 2. Pick an IDLE NPU (the script refuses to run without one)
```bash
npu-smi info    # pick a chip with AICore 0% that also shows
                # "No running processes found in NPU N"
export ASCEND_RT_VISIBLE_DEVICES=N      # e.g. 1
```
The `run` step also enforces this: it aborts if the chosen NPU isn't idle
(0% AICore + no process), unless you pass `SKIP_IDLE_GATE=1`.

### 3. Build + benchmark `main`
```bash
cd "$MAIN_REPO"
git checkout "$MAIN_BRANCH" && git pull origin "$MAIN_BRANCH"
git submodule update --init --recursive
bash build.sh -a kernels

export SGL_KERNEL_SO="$MAIN_REPO/$SO_REL"
python3 "$TOOLS/bench_causal_conv1d_compare.py" run \
    --label main --out "$TOOLS/main.json"
```

### 4. Build + benchmark the branch
```bash
cd "$BRANCH_REPO"
git checkout "$FEAT_BRANCH" && git pull origin "$FEAT_BRANCH"
git submodule update --init --recursive
bash build.sh -a kernels

export SGL_KERNEL_SO="$BRANCH_REPO/$SO_REL"
python3 "$TOOLS/bench_causal_conv1d_compare.py" run \
    --label tiling --out "$TOOLS/tiling.json"
```

### 5. Compare (no NPU needed)
```bash
python3 "$TOOLS/bench_causal_conv1d_compare.py" compare \
    "$TOOLS/main.json" "$TOOLS/tiling.json" \
    --csv "$TOOLS/cmp_main_vs_tiling.csv"
```
Table columns: `main_e2e`/`tiling_e2e` = host latency (µs),
`main_kernel`/`tiling_kernel` = on-device kernel time (µs), and
`speedup_e2e`/`speedup_kernel` = `main / tiling` per shape (**>1 ⇒ tiling faster**).
The summary (median, geomean, wins) prints to **stderr**, separately for E2E and KERNEL.

### Notes / gotchas
- **Rebuild after every `git checkout`** — the `.so` is a build artifact. Each run
  prints `git=<branch>@<commit> md5=<...>`; if the md5 didn't change after a
  rebuild, you're benchmarking stale code.
- **Same terminal** for steps 0–5 (the `$MAIN_REPO` etc. variables are shell-local).
- **Default grid** (per build): seq {128,256,512,1024} × dim {1024,2048,4096,5120,6144}
  × batch {1,3,4,6,12,16,32,64,128,256} × {3d,varlen} = 400 shapes. It's thorough,
  so a full sweep takes a while — narrow it for a quick pass (see below).
- Tuning knobs (prefix any `run`):
  - grid: `BATCHES=1,6,32,128 DIMS=2048,4096 SEQS=512 LAYOUTS=3d,varlen`
  - iters: `E2E_ITERS=100 E2E_WARM=10 KERNEL_N=100 KERNEL_WARM=10`
  - `E2E_ONLY=1` skip kernel profiling (faster) · `QUICK=1` 4-shape smoke test
