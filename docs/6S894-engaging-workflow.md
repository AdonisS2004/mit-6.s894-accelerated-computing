# 6.S894 on Engaging — Lab Workflow (Telerun-free)

A single runbook for completing every lab on MIT's Engaging cluster, compiling and
running directly with `nvcc`/`g++` instead of Telerun. Each lab's source files are
self-contained (own `main()`, arg parser, benchmark harness, correctness check), so
"replacing Telerun" just means compiling the file yourself on a GPU node.

Replace `YOUR_KERBEROS` everywhere with your Kerberos username.

---

## 0. One-time setup (do once, ever)

**a. Activate your Engaging account.** Log into the OnDemand portal once with your
Kerberos credentials so the system provisions your account, then wait a few minutes:

    https://engaging-ood.mit.edu

**b. Add this to `~/.ssh/config` on your laptop.** The `ProxyJump` lets VS Code reach a
compute node through the login node; the `Control*` lines collapse repeated Duo prompts
into one per session.

    Host orcd-login
        HostName orcd-login.mit.edu
        User YOUR_KERBEROS
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlPersist 4h

    Host orcd-compute
        HostName node0000          # update each session to your srun node (step 1)
        User YOUR_KERBEROS
        ProxyJump orcd-login
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlPersist 4h

**c. (Optional) Save a reusable Makefile** one level above where you clone your labs
so you can copy it into any lab with `cp ../lab.mk Makefile`. Create it at that path
(e.g. `~/lab.mk` if you clone labs into `~/labN`):

    ARCH      ?= native
    LDFLAGS   ?=
    INCLUDES  ?=
    NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) $(INCLUDES)
    CXXFLAGS  := -O3 -std=c++17 -march=native

    CU_BINS  := $(patsubst %.cu,%,$(wildcard *.cu))
    CPP_BINS := $(patsubst %.cpp,%,$(wildcard *.cpp))

    all: out $(CU_BINS) $(CPP_BINS)
    out: ; mkdir -p out
    %: %.cu  ; nvcc $(NVCCFLAGS) $(LDFLAGS) $< -o $@
    %: %.cpp ; g++  $(CXXFLAGS) $< -o $@
    clean:   ; rm -f $(CU_BINS) $(CPP_BINS)

`make` builds every `.cu`/`.cpp` in the folder and auto-creates `out/`. Override flags
per lab — for Hopper (labs 9–10): `make ARCH=sm_90a LDFLAGS="-lcuda -lcublas"`.

---

## 1. Start a work session (every time you sit down)

Open a terminal and grab an interactive GPU node. **Keep this shell open** — it holds
your Slurm allocation, which is also what lets VS Code SSH into the node.

    ssh orcd-login
    srun -p mit_normal_gpu -G 1 --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash

Once you land on the node:

    hostname              # e.g. node3005  — note this
    module load deprecated-modules
    module load cuda/12.4.0
    nvcc --version        # confirm toolkit
    nvidia-smi            # confirm GPU + see which card you got
    lscpu | grep avx512f  # confirm AVX-512 (needed for the CPU lab files)

`cuda/12.4.0` lives under the `deprecated-modules` namespace on Engaging, so you must
load that first or Lmod will report the module as unavailable. If you forget, run
`module spider cuda/12.4.0` and it will tell you the same thing.

Expected output if everything is working:

    [adonisse@node3005 ~]$ module load deprecated-modules
    [adonisse@node3005 ~]$ module load cuda/12.4.0
    [adonisse@node3005 ~]$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Tue_Feb_27_16:19:38_PST_2024
    Cuda compilation tools, release 12.4, V12.4.99

    [adonisse@node3005 ~]$ nvidia-smi
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 590.48.01    Driver Version: 590.48.01      CUDA Version: 13.1              |
    +-----------------------------------------+------------------------+----------------------+
    |   0  NVIDIA L40S                    On  |   00000000:E1:00.0 Off |                    0 |
    |  34W / 350W                             |       0MiB /  46068MiB |      0%      Default |
    +-----------------------------------------------------------------------------------------+

    [adonisse@node3005 ~]$ lscpu | grep avx512f
    Flags: ... avx512f avx512dq ... avx512bw avx512vl ...  # long line, avx512f confirms AVX-512

Then point VS Code at the node: edit the `orcd-compute` `HostName` in `~/.ssh/config`
to the `hostname` above, and in VS Code run **Remote-SSH: Connect to Host → orcd-compute**.
Your editor is now local; compiles and runs happen on the GPU node.

> If `lscpu | grep avx512f` prints nothing, that node's CPU lacks AVX-512 and the
> `*_cpu.cpp` file won't run. Either re-`srun` for a different node or build/run just the
> CPU file on an `mit_normal` node that has it.

---

## 2. Get the lab

Each lab is its own public repo, so plain HTTPS works (no GitHub key needed):

    git clone https://github.com/accelerated-computing-class/labN.git
    cd labN
    cp ../lab.mk Makefile     # if you saved the reusable Makefile

---

## 3. Build and run

### The inner loop
    make                      # builds all .cu/.cpp and creates out/
    ./<gpu_binary>            # runs all impls, writes out/*.bmp, prints correctness
    ./<cpu_binary>

### Explicit commands (if you skip the Makefile)
    # GPU file (CUDA) — labs 1–8
    nvcc -O3 -std=c++17 -arch=sm_89 <file>.cu -o <file>
    # GPU file (CUDA) — labs 9–10 (Hopper-specific features + extra libs)
    nvcc -O3 -std=c++17 -arch=sm_90a -lcuda -lcublas <file>.cu -o <file>
    # CPU file (AVX-512)
    g++  -O3 -std=c++17 -march=native <file>.cpp -o <file>

### Common harness flags (Lab 1; most labs follow this pattern)
    ./mandelbrot_gpu                      # defaults (256px, 1000 iters, all impls)
    ./mandelbrot_gpu -i scalar -r 512 -b 1000   # -i impl, -r resolution, -b max_iters

The harness prints `average output difference from reference` (0 = correct) and dumps
`.bmp` files to `out/`. On a fresh clone the difference is large because the kernels are
empty `// TODO` stubs — that's the harness working, not a setup error.

### GPU architecture per lab
| Labs  | GPU              | nvcc flag      | make shortcut                          |
|-------|------------------|----------------|----------------------------------------|
| 1–8   | L40S (Ada)       | `-arch=sm_89`  | `make ARCH=sm_89`                      |
| 9–10  | H100 (Hopper)    | `-arch=sm_90a` | `make ARCH=sm_90a LDFLAGS="-lcuda -lcublas"` |

`sm_90a` (not plain `sm_90`) enables the Hopper-specific features (TMA, wgmma) the later
labs use. `-arch=native` resolves to `sm_90`, so set `sm_90a` explicitly for those.

### Profiling (labs 2+)
Once correctness is green, use these to find bottlenecks and hit the performance thresholds:

    ncu --set full ./<binary>    # Nsight Compute: roofline, memory throughput, occupancy
    nsys profile ./<binary>      # Nsight Systems: full timeline view

### Viewing result images
In VS Code Remote, just open `out/*.bmp`. Or copy down (ProxyJump makes this work):

    scp orcd-compute:~/labN/out/<name>.bmp .

---

## 4. Hopper labs (9–10) via batch

Interactive H100s can be scarce, so submit these as batch jobs. First check what's
available and on which partition:

    sinfo -p mit_normal_gpu -N -o "%N %G"   # GPU types in this partition
    # if H100 isn't here, also try:  sinfo -p mit_preemptable -N -o "%N %G"

Create `run.sh` in the lab folder:

    #!/bin/bash
    #SBATCH -p mit_normal_gpu       # or the partition where H100 lives
    #SBATCH -G h100:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=01:00:00
    #SBATCH -o slurm-%j.out

    module load deprecated-modules
    module load cuda/12.4.0
    make ARCH=sm_90a LDFLAGS="-lcuda -lcublas"
    ./<file>

Then:

    sbatch run.sh         # submit
    squeue --me           # watch the queue
    cat slurm-<jobid>.out # results when it finishes

---

## 5. Lab 11 (JAX / Pallas / TPU)

Lab 11 is Python-based and targets TPU via JAX Pallas — not a CUDA/GPU lab. The workflow
is different:

**Request a TPU node** (or the partition that has TPUs on Engaging):

    srun -p <tpu_partition> --time=01:00:00 --pty bash

**Set up the Python environment** (once per cluster account):

    python3 -m venv ~/venv-jax
    source ~/venv-jax/bin/activate
    pip install jax[tpu] numpy

**Each session:**

    source ~/venv-jax/bin/activate
    cd lab11

**Run:**

    python collective_matmul.py
    python collectives.py

Set `ENABLE_DEBUG = True` inside `collective_matmul.py` to enable `jax.debug.print`,
out-of-bounds detection, and race-condition detection (at a performance cost). Set it
back to `False` before timing runs.

---

## 6. Finish a lab

A lab is "done" when:
1. The correctness difference reads ~0 for every implementation you were asked to write.
2. (Labs 2+) Your timings beat the lab's stated performance thresholds (found at
   `https://accelerated-computing-class.github.io/fall24/labs/labN`).
3. You've answered the lab's write-up questions (the `Question N for write-up` prompts).

Deliverables are the edited source files plus a short PDF of your answers. If you're
enrolled, submit both to Gradescope; if you're self-studying, the correctness check +
your written answers are the real completion signal.

---

## 7. End the session (don't strand a GPU)

    exit          # leaves the compute node and releases the GPU allocation

If you backgrounded a job or it's misbehaving:

    squeue --me
    scancel <jobid>

---

## 8. Troubleshooting quick-reference

| Symptom | Fix |
|---|---|
| `nvcc: command not found` | `module load deprecated-modules && module load cuda/12.4.0` |
| `cuda/12.4.0` cannot be loaded / module unavailable | load `deprecated-modules` first; Engaging requires it as a prerequisite |
| Runtime "no kernel image is available" | `-arch` doesn't match your GPU — check `nvidia-smi`, rebuild with the right `sm_` |
| `*_cpu.cpp` won't compile (`_mm512_*` errors) | node lacks AVX-512 — `lscpu | grep avx512f`; use a different node |
| `*_cpu` binary dies with illegal instruction | compiled with AVX-512 but ran on a node without it — build and run on the same node |
| Duo prompts on every connect | confirm the `Control*` lines are in `~/.ssh/config` |
| Long queue wait for `srun` | lower `--time`/`--mem`, or try `-p mit_preemptable` |
| VS Code "permission denied" to compute node | make sure the `srun` shell (allocation) is still open on that node |
| labs 9–10 link errors (`-lcuda`, `-lcublas`) | use `make ARCH=sm_90a LDFLAGS="-lcuda -lcublas"` or add flags to `run.sh` |
| Lab 11 `jax` import error | activate the venv: `source ~/venv-jax/bin/activate` |

---

### The whole loop, condensed
    ssh orcd-login
    srun -p mit_normal_gpu -G 1 --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash
    module load deprecated-modules && module load cuda/12.4.0
    git clone https://github.com/accelerated-computing-class/labN.git && cd labN
    cp ../lab.mk Makefile
    # ...edit in VS Code Remote...
    make && ./<binary>
    exit
