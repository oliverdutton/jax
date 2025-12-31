#!/bin/bash

# Test with enhanced LD_PRELOAD that intercepts more syscalls

echo "================================================================================"
echo "TESTING WITH ENHANCED LD_PRELOAD"
echo "================================================================================"

export LD_PRELOAD=/home/user/jax/tpu_bypass_enhanced.so

# TPU environment
export TPU_CHIPS_PER_HOST_BOUNDS='1,1,1'
export TPU_HOST_BOUNDS='1,1,1'
export TPU_CHIPS='1'
export TPU_LOAD_LIBRARY='1'
export CLOUD_TPU_TASK_ID='0'
export TPU_WORKER_ID='0'
export TPU_WORKER_HOSTNAMES='localhost'
export TPU_ACCELERATOR_TYPE='v5litepod-1'

# Create dump directory
DUMP_DIR=$(mktemp -d -t tpu_llo_enhanced_XXXXX)

# XLA flags
export XLA_FLAGS="--xla_dump_to=$DUMP_DIR --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"

# LIBTPU flags for LLO dumps
export LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=true --xla_jf_dump_llo_proto=true --xla_jf_dump_hlo_text=true --xla_jf_dump_llo_pass_label_regex=.*"

# JAX platforms
export JAX_PLATFORMS='cpu,tpu'

echo ""
echo "Environment:"
echo "  LD_PRELOAD=$LD_PRELOAD"
echo "  DUMP_DIR=$DUMP_DIR"
echo ""

# Run Python with timeout (in case it hangs)
timeout 30 python3 << 'PYTHON_EOF'
import os
import sys
import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

print("="*80)
print("TESTING TPU INITIALIZATION")
print("="*80)

print(f"\nJAX version: {jax.__version__}")

try:
    print("\nAttempting to get JAX devices...")
    devices = jax.devices()
    print(f"✓ JAX devices: {devices}")

    tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
    if tpu_devices:
        print(f"\n✓✓✓ TPU BACKEND INITIALIZED! ✓✓✓")
        print(f"TPU devices: {tpu_devices}")
    else:
        print("\n✗ No TPU devices found")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("COMPILING WITH MOCKED TPU v5e")
print("="*80)

tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

def mini_attention(x, w1, w2):
    h = x @ w1
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    h = h / rms
    h_max = jnp.max(h, axis=-1, keepdims=True)
    exp_h = jnp.exp(h - h_max)
    h = exp_h / jnp.sum(exp_h, axis=-1, keepdims=True)
    out = h @ w2
    return out

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
x = jax.random.normal(k1, (16, 64))
w1 = jax.random.normal(k2, (64, 64)) * 0.02
w2 = jax.random.normal(k3, (64, 32)) * 0.02

print("\nLowering...")
with use_abstract_mesh(mesh):
    lowered = jax.jit(mini_attention).lower(x, w1, w2)
    print("✓ Lowered")

    print("\nCompiling...")
    compiled = lowered.compile()
    print("✓ Compiled")

print("\n✓ Test completed successfully")

PYTHON_EOF

PYTHON_EXIT=$?
echo ""
echo "Python exit code: $PYTHON_EXIT"

echo ""
echo "================================================================================"
echo "CHECKING FOR LLO DUMPS"
echo "================================================================================"
echo "Dump directory: $DUMP_DIR"
echo ""

LLO_COUNT=$(find "$DUMP_DIR" -name "*llo*" 2>/dev/null | wc -l)
HLO_COUNT=$(find "$DUMP_DIR" -name "*hlo*" 2>/dev/null | wc -l)

echo "LLO files: $LLO_COUNT"
echo "HLO files: $HLO_COUNT"

if [ "$LLO_COUNT" -gt 0 ]; then
    echo ""
    echo "✓✓✓ LLO FILES FOUND! ✓✓✓"
    echo ""
    echo "LLO files:"
    find "$DUMP_DIR" -name "*llo*" -type f | head -10

    # Display first LLO file content
    FIRST_LLO=$(find "$DUMP_DIR" -name "*llo*" -type f | head -1)
    if [ -n "$FIRST_LLO" ]; then
        echo ""
        echo "================================================================================"
        echo "FIRST LLO FILE: $(basename $FIRST_LLO)"
        echo "================================================================================"
        head -100 "$FIRST_LLO"
        echo ""
        echo "... ($(wc -l < "$FIRST_LLO") total lines)"
    fi
else
    echo ""
    echo "✗ No LLO files found"
fi

echo ""
echo "All dump files:"
find "$DUMP_DIR" -type f | head -20

echo ""
echo "================================================================================"
echo "Dump directory: $DUMP_DIR"
echo "================================================================================"
