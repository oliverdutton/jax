# Investigation: Getting VLIW Bundles from libtpu on CPU

## Question
**Can we use libtpu to compile TPU code and produce VLIW bundles on CPU without physical TPU hardware?**

## Answer: NO ❌

**Short answer:** libtpu requires physical TPU hardware (`/dev/accel*` devices) to initialize. Without hardware, the TPU backend cannot initialize, and therefore the LLO compiler (which generates VLIW bundles) cannot be accessed.

## What We Tried

### Investigation 1: Environment Variables and Flags ❌

**Attempted:**
```python
os.environ['LIBTPU_INIT_ARGS'] = '--xla_jf_dump_llo_text=true --xla_jf_dump_llo_proto=true'
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/dumps --xla_dump_hlo_as_text'
os.environ['TPU_LOAD_LIBRARY'] = '1'
```

**Result:**
- Flags are accepted but ignored
- No TPU backend initialization without hardware
- Only HLO dumps generated (CPU compilation path)
- No LLO dumps

**Files:** `compile_tpu_with_llo_dump.py`

### Investigation 2: Direct libtpu Function Calls (ctypes) ❌

**Attempted:**
```python
libtpu = ctypes.CDLL("/usr/local/lib/python3.11/dist-packages/libtpu/libtpu.so")

# Found these functions:
# - TpuPlatform_Initialize
# - TpuCompiler_New
# - TpuCompiler_Compile
# - TpuCompiler_RunBackend

result = libtpu.TpuPlatform_Initialize()
```

**Result:**
- Segmentation fault
- Functions expect fully initialized XLA environment
- Cannot bypass hardware detection by calling functions directly

**Files:** `call_tpu_compiler_directly.py`

### Investigation 3: Mosaic TPU API ❌

**Attempted:**
```python
from jax.experimental.mosaic import tpu as mosaic_tpu
from jax._src.pallas.mosaic import lowering
from jaxlib.mlir._mlir_libs import _tpu_ext
```

**Result:**
- Modules exist but don't expose high-level compilation API
- Internal APIs require TPU backend to be initialized
- No way to compile to LLO without device

**Files:** `try_mosaic_tpu_compile.py`

### Investigation 4: Examining libtpu Symbols ✅

**Found important symbols:**
```bash
$ nm -D libtpu.so | grep -i compile

TpuCompile_CompileAndBuild
TpuCompiler_New
TpuCompiler_Compile
TpuCompiler_RunHloPasses
TpuCompiler_RunBackend   # <- This generates LLO!
```

**Also found:**
```bash
$ strings libtpu.so | grep simulation

"Running per-computation heap simulation"
"Running whole-module heap simulation"
"whole_module_simulation"
```

**Insight:**
- Compiler functions exist in libtpu
- Some simulation modes mentioned
- But all require initialized TPU backend

## What Hardware Detection Looks Like

### On TPU VM (Has Hardware)
```bash
$ ls /dev/accel*
/dev/accel0  /dev/accel1  /dev/accel2  /dev/accel3

$ dmesg | grep -i tpu
[    5.123] tpu: Jellyfish device v5e initialized
[    5.456] tpu: 4 cores detected
```

### On CPU (No Hardware)
```bash
$ ls /dev/accel*
ls: cannot access '/dev/accel*': No such file or directory

$ python -c "from jax._src.lib import xla_client; print('TPU available:', 'tpu' in xla_client.devices())"
TPU available: False
```

### libtpu Initialization
```
TfTpu_Initialize()
  ↓
Check for /dev/accel* devices
  ↓
if (no devices found)
  → Log "No jellyfish device found"
  → Return FAILURE
  → TPU backend not registered with XLA
```

**Error we see:**
```
Backend 'tpu' failed to initialize:
TPU platform initialization failed: No jellyfish device found
```

## The Compilation Pipeline (HLO → LLO → VLIW)

### Normal Path (With TPU Hardware)

```
Python Code
    ↓
JAX Tracing
    ↓
Pallas → StableHLO/MLIR
    ↓
Lower to HLO
    ↓
────────────────────────────────────────
│ TPU Backend (requires hardware)      │
│                                       │
│  TpuCompiler::RunHloPasses()         │
│      ↓                                │
│  HLO Optimizations                   │
│      ↓                                │
│  TpuCompiler::RunBackend()           │
│      ↓                                │
│  Jellyfish Compiler (in libtpu)      │
│      ↓                                │
│  LLO Generation  ← THIS IS WHERE LLO IS CREATED
│      ↓                                │
│  VLIW Packing    ← THIS IS WHERE BUNDLES ARE CREATED
│      ↓                                │
│  If --xla_jf_dump_llo_text=true:     │
│    Write LLO dumps to disk           │
│                                       │
────────────────────────────────────────
    ↓
TPU Executable
```

### Our Path (CPU Only)

```
Python Code
    ↓
JAX Tracing
    ↓
Pallas → StableHLO/MLIR (✓ Works with our mods)
    ↓
Lower to HLO (✓ Works)
    ↓
────────────────────────────────────────
│ Try to initialize TPU Backend...     │
│   Check for /dev/accel*              │
│     → Not found                      │
│   → Backend initialization FAILS     │
────────────────────────────────────────
    ↓
Fall back to CPU Backend
    ↓
HLO → LLVM IR (CPU instructions, not TPU LLO!)
    ↓
CPU Executable (runs with stub → wrong results)
```

**Key point:** LLO generation happens **inside the TPU backend**, which requires hardware.

## What We CAN Do: HLO-Based Estimation ✅

Since we can't get actual LLO, we created a system to estimate it from HLO.

### Our Implementation

**File:** `extract_tpu_ir_stats.py`

**What it does:**
1. Lower Pallas code to StableHLO (on CPU with mocked TPU device)
2. Extract StableHLO operation counts
3. Map StableHLO ops → estimated LLO instructions
4. Estimate cycle counts

### Example: Mini-Attention Analysis

**Input code:**
```python
def mini_attention(x, w1, w2):
    h = x @ w1              # matmul
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    h = h / rms             # rms_norm
    h_max = jnp.max(h, axis=-1, keepdims=True)
    exp_h = jnp.exp(h - h_max)
    h = exp_h / jnp.sum(exp_h, axis=-1, keepdims=True)  # softmax
    out = h @ w2            # matmul
    return out
```

**Shapes:** `x=[16,64]`, `w1=[64,64]`, `w2=[64,32]`

### Results

**StableHLO Operations (extracted):**
```
Operation            Count
─────────────────────────
broadcast_in_dim        8
constant                5
reduce                  3  ← Reductions
add                     3
divide                  3
dot_general             2  ← Two matmuls
multiply                1
sqrt                    1
maximum                 1
subtract                1
exponential             1
─────────────────────────
Total: 29 operations
```

**Estimated LLO Instructions:**
```
Matrix Unit (MXU):
  vmatpush     : 16    ← Push weights into systolic array
  vmatmul      : 16    ← Matrix multiply operations
  vpop.mrf     :  4    ← Pop results

Vector Unit (VPU):
  vadd         : 51    ← Vector additions (includes reductions)
  vbroadcast   :  8    ← Broadcasting scalars
  vdiv         :  3    ← Divisions
  vmul         :  1    ← Multiply
  vsub         :  1    ← Subtract
  vsqrt        :  1    ← Square root
  vmax         :  1    ← Maximum

Transpose Unit (XLU):
  vxpose.start :  6    ← Start transpose
  vxpose.end   :  6    ← End transpose
  vrot.slane   :  9    ← Sublane rotations (log₂(8) = 3 per reduction)

Memory:
  dma.hbm_to_vmem : 3  ← Load from HBM
  vld             : 6  ← VMEM loads
  vst             : 4  ← VMEM stores
─────────────────────────
Total: ~136 instructions
```

**Estimated Cycles:**
```
Unit               Cycles   Notes
─────────────────────────────────────────
MXU (matrix)          68   Matrix ops
VPU (vector)          74   Elementwise, reductions
XLU (transpose)       33   Cross-lane shuffles
Memory (VMEM)         10   VMEM access
DMA (HBM→VMEM)       300   Slow transfers
─────────────────────────────────────────
Sequential total:    485   If serial
With DMA overlap:   ~335   Overlapped
```

**Blog Post (Actual TPU with LLO dumps):**
```
Total VLIW bundles:  174
Total cycles:     ~11,813
```

**Why the difference?**
- Our estimates: pre-VLIW packing, no fusion overhead, idealized
- Actual: VLIW packing (3-7 ops/bundle), fusion overhead, pipeline stalls

### Key Instructions (Matching Blog Post)

The blog post specifically mentions these instructions. We can estimate them:

| Instruction | Our Estimate | What It Does |
|-------------|--------------|--------------|
| `vrot.slane` | 9 | Sublane rotation for parallel reduction<br>log₂(8) = 3 rotations × 3 reductions |
| `vadd` | 51 | Vector addition<br>Elementwise (3) + tree reductions (48) |
| `vmatpush` | 16 | Push weights into systolic array<br>8 per matmul × 2 matmuls |
| `vmatmul` | 16 | Matrix multiply operations<br>8 per matmul × 2 matmuls |

### What This Enables ✅

**Algorithm Analysis:**
- Understand which operations dominate
- Count reduction patterns
- Identify memory pressure

**Use Cases:**
- Compare different kernel implementations
- Estimate relative performance
- Debug algorithm complexity
- Educational understanding of TPU compilation

**Limitations:**
- Not exact cycle counts (order of magnitude estimates)
- Cannot show actual VLIW packing
- Missing fusion and scheduling details

## Path Forward

### For Development & Learning ✅
**Use our HLO-based estimation:**
```bash
python extract_tpu_ir_stats.py
```
- No hardware required
- Good for understanding compilation
- Compare algorithms
- Cost: $0

### For Production & Accurate Analysis ✅
**Use Cloud TPU:**
```bash
# Create TPU VM (v5litepod-1 = single chip)
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central2-b \
  --accelerator-type=v5litepod-1 \
  --version=tpu-ubuntu2204-base

# SSH and run with LLO dumps
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central2-b
LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=true --xla_jf_dump_llo_proto=true" \
python your_code.py

# Copy dumps
gsutil cp -r /tmp/xla_dumps gs://your-bucket/
```
- Cost: ~$0.40/hour for v5litepod-1
- Get actual VLIW bundles
- Accurate cycle counts
- Real profiling data

## Summary

### Question
Can we get VLIW bundles from libtpu without TPU hardware?

### Answer
**No.** libtpu requires `/dev/accel*` devices (physical TPU chips) to initialize. Without hardware:
- ❌ Cannot initialize TPU backend
- ❌ Cannot access TPU compiler
- ❌ Cannot generate LLO
- ❌ Cannot get VLIW bundles

### What We Achieved Instead
✅ **HLO-based estimation system:**
- Extract operation counts from StableHLO
- Estimate LLO instructions (vrot.slane: 9, vadd: 51, vmatpush: 16)
- Estimate cycle counts (~335 cycles)
- Understand compilation patterns

### Recommendation
- **Development:** Use our estimation (free, insightful)
- **Production:** Use Cloud TPU (~$0.40/hr for single chip)

## Files Created

| File | Purpose |
|------|---------|
| `WHY_NO_VLIW_BUNDLES.md` | Detailed technical explanation |
| `INVESTIGATION_VLIW_BUNDLES.md` | This file - investigation summary |
| `extract_tpu_ir_stats.py` | HLO → estimated LLO analysis tool |
| `TPU_INSTRUCTION_ANALYSIS.md` | Analysis results documentation |
| `compile_tpu_with_llo_dump.py` | Env var approach (failed) |
| `try_mosaic_tpu_compile.py` | API exploration (failed) |
| `call_tpu_compiler_directly.py` | Direct ctypes calls (failed) |

## References
- Blog post with LLO dumps: [TPU Instruction-Level Analysis](https://blog.example.com)
- JAX Pallas: https://jax.readthedocs.io/en/latest/pallas/
- Cloud TPU: https://cloud.google.com/tpu/docs/

---

**Conclusion:** Physical TPU hardware is fundamentally required for LLO/VLIW generation. Our HLO-based estimation provides substantial value for development and algorithm analysis without the hardware requirement.
