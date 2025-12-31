#!/bin/bash

rm -f /tmp/tpu_aggressive.log

export LD_PRELOAD=/home/user/jax/tpu_aggressive_emulator.so

# Force everything
export JAX_FORCE_TPU_INIT=1
export TPU_CHIPS='1'
export TPU_LOAD_LIBRARY='1'
export TPU_ACCELERATOR_TYPE='v5litepod-1'
export JAX_PLATFORMS='cpu,tpu'

DUMP_DIR=$(mktemp -d -t tpu_aggressive_XXXXX)
export XLA_FLAGS="--xla_dump_to=$DUMP_DIR --xla_dump_hlo_as_text"
export LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=true --xla_jf_dump_llo_proto=true"

echo "================================================================================"
echo "AGGRESSIVE EMULATOR TEST"
echo "================================================================================"
echo "LD_PRELOAD=$LD_PRELOAD"
echo "DUMP_DIR=$DUMP_DIR"
echo ""

timeout 60 python3 << 'PYTHON_EOF'
import os
import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

print("Testing TPU initialization...")

try:
    devices = jax.devices()
    print(f"Devices: {devices}")
    
    tpu = [d for d in devices if 'tpu' in str(d).lower()]
    if tpu:
        print(f"✓✓✓ TPU FOUND: {tpu}")
    else:
        print("No TPU devices")
except Exception as e:
    print(f"Error: {e}")

print("\nCompiling...")
tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

def f(x, w):
    return x @ w

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (16, 64))
w = jax.random.normal(key, (64, 32))

with use_abstract_mesh(mesh):
    lowered = jax.jit(f).lower(x, w)
    compiled = lowered.compile()
    print("✓ Compiled")

PYTHON_EOF

echo ""
echo "================================================================================"
echo "CHECKING LOGS"
echo "================================================================================"

if [ -f /tmp/tpu_aggressive.log ]; then
    echo "Emulator log:"
    grep -E "AGGRESSIVE.*accel|AGGRESSIVE.*tpu|AGGRESSIVE.*sys" /tmp/tpu_aggressive.log | head -50
    
    echo ""
    echo "Total intercepted calls:"
    grep "\[AGGRESSIVE\]" /tmp/tpu_aggressive.log | grep -v "Shutting\|EMULATOR LOADED" | wc -l
else
    echo "No log file!"
fi

echo ""
echo "Checking for LLO dumps..."
find "$DUMP_DIR" -name "*llo*" -type f | wc -l

echo "================================================================================"
