"""
Extract instruction-level statistics from TPU IR.

Since we can lower (but not compile) TPU code on CPU, we can analyze
the StableHLO IR to understand what operations would be generated.
"""
import re
from collections import Counter
import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

jax.config.update('jax_platforms', 'cpu')


def mini_attention(x, w1, w2):
    """Mini attention: matmul → rms_norm → softmax → matmul"""
    # matmul_1
    h = x @ w1

    # rms_norm
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    h = h / rms

    # softmax
    h_max = jnp.max(h, axis=-1, keepdims=True)
    exp_h = jnp.exp(h - h_max)
    h = exp_h / jnp.sum(exp_h, axis=-1, keepdims=True)

    # matmul_2
    out = h @ w2

    return out


def parse_stablehlo_operations(ir_text):
    """Extract StableHLO operations from IR text."""
    op_counts = Counter()

    # Match StableHLO operations: stablehlo.operation_name
    stablehlo_pattern = r'stablehlo\.(\w+)'
    matches = re.findall(stablehlo_pattern, ir_text)

    for op in matches:
        op_counts[op] += 1

    return op_counts


def estimate_llo_from_stablehlo(stablehlo_ops):
    """
    Estimate LLO-level instructions based on StableHLO operations.

    This simulates what the TPU backend would generate.
    """
    llo_est = Counter()

    # Dot products become matrix operations
    if 'dot_general' in stablehlo_ops or 'dot' in stablehlo_ops:
        matmul_count = stablehlo_ops.get('dot_general', 0) + stablehlo_ops.get('dot', 0)

        # Each matmul requires (rough estimates for 16x64 @ 64x64):
        # - Load weights: ~8 vmatpush operations (64x64 / 8x128 tiles)
        # - Compute: ~8 vmatmul operations
        # - Extract results: ~2 vpop operations
        llo_est['vmatpush'] = matmul_count * 8
        llo_est['vmatmul'] = matmul_count * 8
        llo_est['vpop.mrf'] = matmul_count * 2

    # Elementwise operations map roughly 1:1 to vector ops
    elementwise_ops = {
        'multiply': 'vmul',
        'add': 'vadd',
        'subtract': 'vsub',
        'divide': 'vdiv',
        'sqrt': 'vsqrt',
        'rsqrt': 'vrsqrt',
        'exp': 'vpow2',  # exp uses pow2 instruction
        'maximum': 'vmax',
    }

    for stablehlo_op, llo_op in elementwise_ops.items():
        if stablehlo_op in stablehlo_ops:
            llo_est[llo_op] += stablehlo_ops[stablehlo_op]

    # Reductions require transpose + tree reduction + sublane rotations
    if 'reduce' in stablehlo_ops:
        reduce_count = stablehlo_ops['reduce']

        # Each reduction:
        # - 2 transpose operations (xpose.start + xpose.end)
        # - ~16 adds for tree reduction (log₂ fold)
        # - 3 rotations for sublane reduction (log₂(8) = 3)
        llo_est['vxpose.start'] = reduce_count * 2
        llo_est['vxpose.end'] = reduce_count * 2
        llo_est['vadd'] += reduce_count * 16  # Tree reduction
        llo_est['vrot.slane'] = reduce_count * 3  # Sublane rotations

    # Broadcasts become vector broadcasts
    if 'broadcast' in stablehlo_ops or 'broadcast_in_dim' in stablehlo_ops:
        broadcast_count = (stablehlo_ops.get('broadcast', 0) +
                          stablehlo_ops.get('broadcast_in_dim', 0))
        llo_est['vbroadcast'] = broadcast_count

    # Memory operations
    # Loads and stores for intermediate results
    llo_est['vld'] = len([op for op in stablehlo_ops if 'reduce' in op or 'dot' in op]) * 3
    llo_est['vst'] = len([op for op in stablehlo_ops if 'reduce' in op or 'dot' in op]) * 2

    # DMA operations (estimated)
    llo_est['dma.hbm_to_vmem'] = 3  # Load x, w1, w2

    return llo_est


def estimate_cycles(llo_ops):
    """Estimate cycle counts based on LLO operations."""

    # Cycle estimates per operation type (rough estimates)
    cycles_per_op = {
        # Matrix operations (MXU)
        'vmatpush': 2,
        'vmatmul': 2,
        'vpop.mrf': 1,

        # Vector operations (VPU) - 1 cycle each
        'vmul': 1,
        'vadd': 1,
        'vsub': 1,
        'vdiv': 3,  # Division is slower
        'vsqrt': 3,
        'vrsqrt': 3,
        'vpow2': 4,  # Exp is expensive
        'vmax': 1,
        'vbroadcast': 1,

        # Transpose/shuffle (XLU)
        'vxpose.start': 2,
        'vxpose.end': 2,
        'vrot.slane': 1,

        # Memory operations
        'vld': 1,
        'vst': 1,
        'dma.hbm_to_vmem': 100,  # DMA is slow
    }

    total_cycles = 0
    breakdown = {}

    for op, count in llo_ops.items():
        cycles = cycles_per_op.get(op, 1) * count
        total_cycles += cycles
        breakdown[op] = cycles

    return total_cycles, breakdown


def main():
    print("="*80)
    print("TPU IR INSTRUCTION-LEVEL ANALYSIS")
    print("="*80)

    # Create mocked TPU v5e device
    print("\nUsing mocked TPU v5e device...")
    tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
    mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

    # Create inputs (same shapes as blog post)
    batch, d_in, d_mid, d_out = 16, 64, 64, 32
    print(f"Shapes: x=[{batch},{d_in}], w1=[{d_in},{d_mid}], w2=[{d_mid},{d_out}]")

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    x = jax.random.normal(k1, (batch, d_in))
    w1 = jax.random.normal(k2, (d_in, d_mid)) * 0.02
    w2 = jax.random.normal(k3, (d_mid, d_out)) * 0.02

    print("\nLowering with TPU v5e device...")

    # Lower with mocked TPU device
    with use_abstract_mesh(mesh):
        jitted_fn = jax.jit(mini_attention)
        lowered = jitted_fn.lower(x, w1, w2)

        print("✓ Lowering succeeded!")

        # Get IR
        ir_text = lowered.as_text()
        print(f"✓ IR length: {len(ir_text)} characters\n")

        # Save IR for inspection
        with open('/home/user/jax/tpu_ir_dump.txt', 'w') as f:
            f.write(ir_text)
        print("✓ IR saved to: /home/user/jax/tpu_ir_dump.txt\n")

    # Parse StableHLO operations
    print("="*80)
    print("STABLEHLO OPERATION COUNTS")
    print("="*80)

    stablehlo_ops = parse_stablehlo_operations(ir_text)

    if not stablehlo_ops:
        print("No StableHLO operations found (IR may be in different format)")
        # Try to extract basic operation counts differently
        print("\nSearching for operation patterns...")

        patterns = {
            'dot/matmul': r'(dot_general|dot|convolution)',
            'multiply': r'multiply',
            'add': r'\badd\b',
            'subtract': r'subtract',
            'divide': r'divide',
            'sqrt': r'sqrt',
            'exp': r'exponential',
            'reduce': r'reduce',
            'broadcast': r'broadcast',
            'max': r'maximum',
        }

        for name, pattern in patterns.items():
            matches = len(re.findall(pattern, ir_text))
            if matches > 0:
                print(f"  {name:20s}: {matches:3d}")
                stablehlo_ops[name] = matches

    else:
        for op, count in sorted(stablehlo_ops.items(), key=lambda x: -x[1]):
            print(f"  {op:25s}: {count:3d}")

    print(f"\nTotal StableHLO operations: {sum(stablehlo_ops.values())}")

    # Estimate LLO instructions
    print("\n" + "="*80)
    print("ESTIMATED LLO INSTRUCTIONS")
    print("="*80)
    print("(This is what the TPU backend would generate)\n")

    llo_ops = estimate_llo_from_stablehlo(stablehlo_ops)

    # Group by hardware unit
    matrix_ops = {k: v for k, v in llo_ops.items()
                  if any(x in k for x in ['vmat', 'vpop.mrf'])}
    vector_ops = {k: v for k, v in llo_ops.items()
                  if k.startswith('v') and k not in matrix_ops and 'vld' not in k and 'vst' not in k}
    xlu_ops = {k: v for k, v in llo_ops.items()
               if 'xpose' in k or 'vrot' in k}
    memory_ops = {k: v for k, v in llo_ops.items()
                  if 'dma' in k or k in ['vld', 'vst']}

    if matrix_ops:
        print("Matrix Unit (MXU) Operations:")
        for op, count in sorted(matrix_ops.items(), key=lambda x: -x[1]):
            print(f"  {op:20s}: {count:4d}")

    if vector_ops:
        print("\nVector Processing Unit (VPU) Operations:")
        for op, count in sorted(vector_ops.items(), key=lambda x: -x[1]):
            print(f"  {op:20s}: {count:4d}")

    if xlu_ops:
        print("\nTranspose Unit (XLU) Operations:")
        for op, count in sorted(xlu_ops.items(), key=lambda x: -x[1]):
            print(f"  {op:20s}: {count:4d}")

    if memory_ops:
        print("\nMemory Operations:")
        for op, count in sorted(memory_ops.items(), key=lambda x: -x[1]):
            print(f"  {op:20s}: {count:4d}")

    total_instructions = sum(llo_ops.values())
    print(f"\nTotal estimated LLO instructions: {total_instructions}")

    # Estimate cycles
    print("\n" + "="*80)
    print("ESTIMATED CYCLE COUNTS")
    print("="*80)

    total_cycles, cycle_breakdown = estimate_cycles(llo_ops)

    # Group by unit
    mxu_cycles = sum(v for k, v in cycle_breakdown.items()
                     if any(x in k for x in ['vmat', 'vpop.mrf']))
    vpu_cycles = sum(v for k, v in cycle_breakdown.items()
                     if k.startswith('v') and k not in matrix_ops
                     and 'xpose' not in k and 'vrot' not in k
                     and k not in ['vld', 'vst'])
    xlu_cycles = sum(v for k, v in cycle_breakdown.items()
                     if 'xpose' in k or 'vrot' in k)
    dma_cycles = sum(v for k, v in cycle_breakdown.items()
                     if 'dma' in k)
    mem_cycles = sum(v for k, v in cycle_breakdown.items()
                     if k in ['vld', 'vst'])

    print(f"\nCycle Breakdown by Unit:")
    print(f"  MXU (matrix):        {mxu_cycles:6d} cycles")
    print(f"  VPU (vector):        {vpu_cycles:6d} cycles")
    print(f"  XLU (transpose):     {xlu_cycles:6d} cycles")
    print(f"  Memory (VMEM):       {mem_cycles:6d} cycles")
    print(f"  DMA (HBM→VMEM):      {dma_cycles:6d} cycles")
    print(f"  " + "-" * 30)
    print(f"  Sequential total:    {total_cycles:6d} cycles")

    # With overlap (DMA can run in parallel)
    overlapped_estimate = total_cycles - (dma_cycles // 2)
    print(f"  With DMA overlap:   ~{overlapped_estimate:6d} cycles")

    print("\n" + "="*80)
    print("COMPARISON WITH BLOG POST")
    print("="*80)
    print("\nBlog post (actual TPU backend with LLO dumps):")
    print("  multiply_reduce_fusion:  71 bundles,  2248 cycles")
    print("  add_sqrt_fusion:         10 bundles,  2120 cycles")
    print("  fusion.5:                56 bundles,  2143 cycles")
    print("  fusion.2:                65 bundles,  2162 cycles")
    print("  fusion (matmul_2):       48 bundles,  3140 cycles")
    print("  Total TLP:              174 bundles, ~11813 cycles")

    print("\nOur estimate (from StableHLO on CPU):")
    print(f"  Total instructions:    ~{total_instructions} (vs 174 bundles)")
    print(f"  Estimated cycles:      ~{overlapped_estimate} (vs ~11813 actual)")

    print("\nNote: Our estimates are rough approximations.")
    print("Actual TPU backend would:")
    print("  • Pack operations into VLIW bundles (multiple ops/cycle)")
    print("  • Apply sophisticated scheduling")
    print("  • Optimize memory access patterns")
    print("  • Fuse operations more aggressively")

    print("\nKey instructions mentioned in blog post:")
    print("  • vrot.slane: rotation for sublane reduction")
    print(f"    Estimated count: {llo_ops.get('vrot.slane', 0)}")
    print("  • vadd: vector addition")
    print(f"    Estimated count: {llo_ops.get('vadd', 0)}")
    print("  • vmatpush/vmatmul: matrix operations")
    print(f"    vmatpush: {llo_ops.get('vmatpush', 0)}, vmatmul: {llo_ops.get('vmatmul', 0)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
