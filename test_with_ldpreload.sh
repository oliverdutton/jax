#!/bin/bash

# Test TPU initialization with LD_PRELOAD to bypass hardware detection

echo "================================================================================"
echo "TESTING TPU WITH LD_PRELOAD BYPASS"
echo "================================================================================"

export LD_PRELOAD=/home/user/jax/tpu_bypass.so

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
DUMP_DIR=$(mktemp -d -t tpu_llo_ldpreload_XXXXX)

# XLA flags for dumps
export XLA_FLAGS="--xla_dump_to=$DUMP_DIR --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"

# LIBTPU flags for LLO dumps
export LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=true --xla_jf_dump_llo_proto=true --xla_jf_dump_hlo_text=true"

# JAX platforms
export JAX_PLATFORMS='cpu,tpu'

echo ""
echo "Environment:"
echo "  LD_PRELOAD=$LD_PRELOAD"
echo "  DUMP_DIR=$DUMP_DIR"
echo "  LIBTPU_INIT_ARGS=$LIBTPU_INIT_ARGS"
echo ""

# Run Python test
python3 << 'PYTHON_EOF'
import os
import sys
import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

print("="*80)
print("PYTHON TEST WITH LD_PRELOAD")
print("="*80)

print(f"\nJAX version: {jax.__version__}")

try:
    print("\nAttempting to get JAX devices...")
    devices = jax.devices()
    print(f"✓ JAX devices: {devices}")

    tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
    if tpu_devices:
        print(f"✓✓✓ TPU DEVICES FOUND: {tpu_devices} ✓✓✓")
    else:
        print("✗ No TPU devices (LD_PRELOAD might not be working)")

except Exception as e:
    print(f"✗ Error getting devices: {e}")
    print("\nThis is expected - LD_PRELOAD intercepts open() but ioctl() needs proper responses")

print("\n" + "="*80)
print("COMPILING WITH MOCKED TPU v5e")
print("="*80)

# Even without real TPU backend, try with mocked device
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

print("\n" + "="*80)
print("DONE")
print("="*80)

PYTHON_EOF

echo ""
echo "================================================================================"
echo "CHECKING DUMP DIRECTORY"
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
    find "$DUMP_DIR" -name "*llo*" -type f | head -10
else
    echo ""
    echo "✗ No LLO files (TPU backend did not compile)"
fi

echo ""
echo "================================================================================"
