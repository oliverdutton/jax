"""
Analyze TPU IR dumps for instruction counts and estimated cycles.

This script uses our CPU lowering capability to generate and analyze
TPU IR for the mini-attention pattern from the blog post.
"""
import os
import re
from collections import Counter, defaultdict

import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

# Force CPU platform
jax.config.update('jax_platforms', 'cpu')

# Create dump directories
DUMP_ROOT = "tpu_analysis/"
HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")
os.makedirs(HLO_DUMP_PATH, exist_ok=True)

# Configure HLO dumps
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_DUMP_PATH} "
    f"--xla_dump_hlo_pass_re=.* "
)


@jax.named_call
def matmul_1(x, w1):
    """Stage 1: Linear projection (like Q @ K^T)"""
    return x @ w1


@jax.named_call
def rms_norm(h):
    """Stage 2: RMS Normalization"""
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    return h / rms


@jax.named_call
def softmax(h):
    """Stage 3: Softmax (row-wise, numerically stable)"""
    h_max = jnp.max(h, axis=-1, keepdims=True)
    exp_h = jnp.exp(h - h_max)
    return exp_h / jnp.sum(exp_h, axis=-1, keepdims=True)


@jax.named_call
def matmul_2(h, w2):
    """Stage 4: Output projection (like attention @ V)"""
    return h @ w2


def mini_attention(x, w1, w2):
    """
    A minimal attention-like block:
    matmul → rms_norm → softmax → matmul
    """
    h = matmul_1(x, w1)
    h = rms_norm(h)
    h = softmax(h)
    out = matmul_2(h, w2)
    return out


def analyze_hlo_ir(dump_path):
    """Analyze HLO IR dumps to extract fusion information and operation counts."""

    print("\n" + "="*80)
    print("HLO IR ANALYSIS")
    print("="*80)

    # Find the final optimized HLO file
    import glob
    hlo_files = glob.glob(os.path.join(dump_path, "*.after_optimizations.txt"))

    if not hlo_files:
        # Try other patterns
        hlo_files = glob.glob(os.path.join(dump_path, "*.txt"))

    if not hlo_files:
        print("No HLO dumps found!")
        return None

    # Use the most recent file
    hlo_file = sorted(hlo_files)[-1]
    print(f"\nAnalyzing: {os.path.basename(hlo_file)}")

    with open(hlo_file, 'r') as f:
        hlo_content = f.read()

    # Extract fusion information
    fusion_pattern = r'%(\w+) = \(.*?\) fusion\((.*?)\), kind=(\w+), calls=(%\w+)'
    fusions = re.findall(fusion_pattern, hlo_content)

    print(f"\n{len(fusions)} Fusions Found:")
    for name, inputs, kind, calls in fusions:
        print(f"  {name}: kind={kind}, calls={calls}")

    # Count operation types
    op_counts = Counter()

    # Common HLO operations
    ops = [
        'dot', 'convolution', 'multiply', 'add', 'subtract', 'divide',
        'sqrt', 'exp', 'log', 'maximum', 'reduce', 'broadcast',
        'reshape', 'transpose', 'copy', 'bitcast'
    ]

    for op in ops:
        pattern = rf'%\w+ = {op}[.\w]*\('
        matches = re.findall(pattern, hlo_content)
        if matches:
            op_counts[op] = len(matches)

    print(f"\nHLO Operation Counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op:15s}: {count:3d}")

    # Extract estimated cycles from backend_config
    cycle_pattern = r'"estimated_cycles":"(\d+)"'
    cycles = re.findall(cycle_pattern, hlo_content)

    if cycles:
        total_cycles = sum(int(c) for c in cycles)
        print(f"\nEstimated Cycles:")
        print(f"  Per fusion: {[int(c) for c in cycles]}")
        print(f"  Total: {total_cycles}")

    # Extract memory space annotations
    vmem_pattern = r'S\(1\)'  # VMEM
    vmem_count = len(re.findall(vmem_pattern, hlo_content))
    print(f"\nMemory Annotations:")
    print(f"  VMEM (S(1)) allocations: {vmem_count}")

    return {
        'fusions': fusions,
        'op_counts': op_counts,
        'cycles': cycles if cycles else []
    }


def extract_ir_operations(ir_text):
    """
    Extract operation counts from IR text.

    Looks for patterns like:
    - vadd, vmul, vrot.slane (vector ops)
    - vmatpush, vmatmul (matrix ops)
    - vld, vst (memory ops)
    - dma operations
    """

    op_counts = Counter()

    # Vector operations
    vector_ops = [
        r'vadd\.\w+',
        r'vmul\.\w+',
        r'vrot\.slane',
        r'vxpose\.\w+',
        r'vpop\.\w+',
        r'vld',
        r'vst',
        r'vsqrt',
        r'vexp',
        r'vdiv',
        r'vsel',
        r'vand',
        r'vcmp',
    ]

    # Matrix operations
    matrix_ops = [
        r'vmatpush\d*\.\w+',
        r'vmatmul\.\w+',
    ]

    # Memory operations
    memory_ops = [
        r'dma\.hbm_to_vmem',
        r'dma\.vmem_to_hbm',
        r'dma\.done\.wait',
    ]

    all_ops = vector_ops + matrix_ops + memory_ops

    for op_pattern in all_ops:
        matches = re.findall(op_pattern, ir_text)
        if matches:
            # Normalize the operation name
            op_name = re.sub(r'\.\w+', '', matches[0])
            op_counts[op_name] += len(matches)

    return op_counts


def simulate_llo_analysis(hlo_analysis):
    """
    Simulate LLO-level analysis based on HLO information.

    Since we don't have actual LLO dumps (would need real TPU backend),
    we estimate what the LLO would contain based on HLO operations.
    """

    print("\n" + "="*80)
    print("SIMULATED LLO ANALYSIS")
    print("="*80)
    print("\n(Note: This is estimated based on HLO. Real LLO would require TPU backend)")

    if not hlo_analysis:
        return

    op_counts = hlo_analysis['op_counts']

    # Estimate LLO instruction counts based on HLO operations
    estimated_llo = defaultdict(int)

    # Matrix operations
    if 'dot' in op_counts or 'convolution' in op_counts:
        matmul_count = op_counts.get('dot', 0) + op_counts.get('convolution', 0)
        # Each matmul typically requires:
        # - Multiple vmatpush for loading weights
        # - vmatmul for computation
        # - vpop for extracting results
        estimated_llo['vmatpush'] = matmul_count * 20  # Rough estimate
        estimated_llo['vmatmul'] = matmul_count * 15
        estimated_llo['vpop.mrf'] = matmul_count * 10

    # Elementwise operations map roughly 1:1
    for op in ['multiply', 'add', 'subtract', 'divide']:
        if op in op_counts:
            estimated_llo[f'v{op[:3]}'] = op_counts[op]

    # Reductions require transpose + tree reduction
    if 'reduce' in op_counts:
        reduce_count = op_counts['reduce']
        estimated_llo['vxpose'] = reduce_count * 2  # start + end
        estimated_llo['vadd'] += reduce_count * 16  # tree reduction
        estimated_llo['vrot.slane'] = reduce_count * 3  # log2(8) sublane rotations

    # Special functions
    if 'sqrt' in op_counts:
        estimated_llo['vrsqrt'] = op_counts['sqrt']
    if 'exp' in op_counts:
        estimated_llo['vpow2'] = op_counts['exp']

    # Memory operations
    estimated_llo['vld'] = len(hlo_analysis['fusions']) * 5
    estimated_llo['vst'] = len(hlo_analysis['fusions']) * 3
    estimated_llo['dma.hbm_to_vmem'] = 3  # Load inputs

    print("\nEstimated LLO Instruction Counts:")
    print("-" * 50)

    total = sum(estimated_llo.values())

    # Group by category
    matrix_ops = {k: v for k, v in estimated_llo.items()
                  if 'vmat' in k or 'vpop.mrf' in k}
    vector_ops = {k: v for k, v in estimated_llo.items()
                  if k.startswith('v') and k not in matrix_ops}
    memory_ops = {k: v for k, v in estimated_llo.items()
                  if 'dma' in k or k in ['vld', 'vst']}

    print("\nMatrix Operations (MXU):")
    for op, count in sorted(matrix_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count:4d}")

    print("\nVector Operations (VPU):")
    for op, count in sorted(vector_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count:4d}")

    print("\nMemory Operations (DMA/VMEM):")
    for op, count in sorted(memory_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count:4d}")

    print(f"\nTotal Estimated Instructions: {total}")

    # Estimate cycle counts
    # These are very rough estimates
    mxu_cycles = matrix_ops.get('vmatmul', 0) * 2  # ~2 cycles per matmul op
    vpu_cycles = sum(vector_ops.values()) * 1      # ~1 cycle per vector op
    memory_cycles = memory_ops.get('dma.hbm_to_vmem', 0) * 100  # DMA is slow

    print(f"\nEstimated Cycle Breakdown:")
    print(f"  MXU (matrix):  ~{mxu_cycles:5d} cycles")
    print(f"  VPU (vector):  ~{vpu_cycles:5d} cycles")
    print(f"  Memory (DMA):  ~{memory_cycles:5d} cycles")
    print(f"  Estimated overlap savings: ~{memory_cycles // 2:5d} cycles")
    print(f"  Net estimate: ~{mxu_cycles + vpu_cycles + memory_cycles // 2:5d} cycles")

    if hlo_analysis['cycles']:
        actual_total = sum(int(c) for c in hlo_analysis['cycles'])
        print(f"  HLO estimate:  {actual_total:5d} cycles")


def main():
    print("="*80)
    print("TPU MINI-ATTENTION COMPILATION ANALYSIS")
    print("="*80)
    print("\nThis analysis uses our CPU lowering capability to inspect")
    print("TPU compilation for the mini-attention pattern from the blog post.")

    # Create mocked TPU v6e device
    print("\nCreating mocked TPU v6e (Trillium) device...")
    tpu_v6e = AbstractDevice(device_kind="TPU v6e", num_cores=1)
    mesh = AbstractMesh((), (), abstract_device=tpu_v6e)

    # Small shapes to keep IR readable (same as blog post)
    batch, d_in, d_mid, d_out = 16, 64, 64, 32

    print(f"Input shapes:")
    print(f"  x:  [{batch}, {d_in}]")
    print(f"  w1: [{d_in}, {d_mid}]")
    print(f"  w2: [{d_mid}, {d_out}]")

    # Create inputs
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    x = jax.random.normal(k1, (batch, d_in))
    w1 = jax.random.normal(k2, (d_in, d_mid)) * 0.02
    w2 = jax.random.normal(k3, (d_mid, d_out)) * 0.02

    print("\nLowering computation with mocked TPU v6e...")

    # Lower with mocked device
    with use_abstract_mesh(mesh):
        jitted_fn = jax.jit(mini_attention)

        # Lower (triggers IR dump)
        lowered = jitted_fn.lower(x, w1, w2)

        print("✓ Lowering succeeded!")

        # Get IR text
        ir_text = lowered.as_text()
        print(f"✓ Generated IR: {len(ir_text)} characters")

    # Analyze the HLO dumps
    hlo_analysis = analyze_hlo_ir(HLO_DUMP_PATH)

    # Simulate LLO analysis
    simulate_llo_analysis(hlo_analysis)

    print("\n" + "="*80)
    print("COMPARISON WITH BLOG POST")
    print("="*80)
    print("\nBlog post reports (with actual TPU backend):")
    print("  - 5 fusions")
    print("  - multiply_reduce_fusion: 71 VLIW bundles, 2248 cycles")
    print("  - add_sqrt_fusion: 10 bundles, 2120 cycles")
    print("  - fusion.5: 56 bundles, 2143 cycles")
    print("  - fusion.2: 65 bundles, 2162 cycles")
    print("  - fusion (matmul_2): 48 bundles, 3140 cycles")
    print("  - Total TLP: 174 bundles")
    print("\nOur analysis (HLO-level on CPU):")
    print(f"  - {len(hlo_analysis['fusions'])} fusions detected in HLO")
    print("  - Estimated instruction counts based on HLO operations")
    print("  - Cannot generate actual VLIW bundles without TPU backend")
    print("\nKey insight: Our CPU lowering allows HLO-level analysis")
    print("without TPU hardware. For actual LLO/VLIW analysis, would need:")
    print("  1. Real TPU backend initialized")
    print("  2. LIBTPU_INIT_ARGS with --xla_jf_dump_llo_text=true")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nHLO dumps written to: {HLO_DUMP_PATH}")
    print("To see full IR: cat " + os.path.join(HLO_DUMP_PATH, "*.txt"))


if __name__ == "__main__":
    main()
