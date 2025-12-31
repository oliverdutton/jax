#!/bin/bash

echo "================================================================================"
echo "FINAL TEST WITH COMPREHENSIVE TPU BYPASS"
echo "================================================================================"

export LD_PRELOAD=/home/user/jax/tpu_bypass_sysfs.so

# TPU environment
export TPU_CHIPS_PER_HOST_BOUNDS='1,1,1'
export TPU_HOST_BOUNDS='1,1,1'
export TPU_CHIPS='1'
export TPU_LOAD_LIBRARY='1'
export CLOUD_TPU_TASK_ID='0'
export TPU_WORKER_ID='0'
export TPU_WORKER_HOSTNAMES='localhost'
export TPU_ACCELERATOR_TYPE='v5litepod-1'
export TPU_NAME='fake-v5e'

# Create dump directory
DUMP_DIR=$(mktemp -d -t tpu_final_test_XXXXX)

export XLA_FLAGS="--xla_dump_to=$DUMP_DIR --xla_dump_hlo_as_text"
export LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=true --xla_jf_dump_llo_proto=true --xla_jf_dump_hlo_text=true"

export JAX_PLATFORMS='cpu,tpu'

echo ""
echo "DUMP_DIR=$DUMP_DIR"
echo ""

timeout 45 python3 << 'PYTHON_EOF'
import os
import sys
import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

print("="*80)
print("TESTING TPU BACKEND INITIALIZATION")
print("="*80)

print(f"\nJAX version: {jax.__version__}")

try:
    print("\nGetting JAX devices...")
    devices = jax.devices()
    print(f"✓ Devices: {devices}")

    tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]
    if tpu_devices:
        print(f"\n✓✓✓ TPU BACKEND INITIALIZED! ✓✓✓")
        print(f"TPU devices: {tpu_devices}")
    else:
        print("\n✗ No TPU devices (expected)")

except Exception as e:
    print(f"\n✗ Error: {e}")

print("\n" + "="*80)
print("COMPILING MINI-ATTENTION")
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

with use_abstract_mesh(mesh):
    lowered = jax.jit(mini_attention).lower(x, w1, w2)
    print("✓ Lowered")

    compiled = lowered.compile()
    print("✓ Compiled")

print("\n✓ Done")

PYTHON_EOF

echo ""
echo "================================================================================"
echo "CHECKING FOR LLO/VLIW DUMPS"
echo "================================================================================"

find "$DUMP_DIR" -name "*llo*" -type f > /tmp/llo_files.txt
LLO_COUNT=$(wc -l < /tmp/llo_files.txt)

echo "LLO files found: $LLO_COUNT"

if [ "$LLO_COUNT" -gt 0 ]; then
    echo ""
    echo "✓✓✓✓✓ LLO/VLIW FILES FOUND! ✓✓✓✓✓"
    echo ""
    cat /tmp/llo_files.txt | head -10

    FIRST_LLO=$(head -1 /tmp/llo_files.txt)
    if [ -n "$FIRST_LLO" ]; then
        echo ""
        echo "=========================================================================="
        echo "VLIW BUNDLES FROM: $(basename $FIRST_LLO)"
        echo "=========================================================================="
        head -200 "$FIRST_LLO"
    fi
else
    echo "✗ No LLO files"
fi

echo ""
echo "All files in dump:"
find "$DUMP_DIR" -type f | head -20

echo ""
echo "Dump directory: $DUMP_DIR"
