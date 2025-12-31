#!/bin/bash

echo "================================================================================"
echo "COMBINED TEST: Fake device + Aggressive LD_PRELOAD + strace"
echo "================================================================================"

# Ensure fake device exists
ls -la /dev/accel0

rm -f /tmp/tpu_aggressive.log /tmp/strace_tpu.log

export LD_PRELOAD=/home/user/jax/tpu_aggressive_emulator.so
export JAX_FORCE_TPU_INIT=1
export TPU_CHIPS='1'
export TPU_LOAD_LIBRARY='1'
export TPU_ACCELERATOR_TYPE='v5litepod-1'
export JAX_PLATFORMS='cpu,tpu'

DUMP_DIR=$(mktemp -d)
export XLA_FLAGS="--xla_dump_to=$DUMP_DIR"
export LIBTPU_INIT_ARGS="--xla_jf_dump_llo_text=true"

echo ""
echo "Running with strace to see what libtpu actually does..."
echo ""

timeout 15 strace -f -e trace=open,openat,stat,lstat,access,ioctl -o /tmp/strace_tpu.log python3 << 'PYTHON_EOF' 2>&1 | head -50
import os
os.environ['JAX_PLATFORMS'] = 'cpu,tpu'

try:
    import jax
    devices = jax.devices()
    print(f"Devices: {devices}")
except Exception as e:
    print(f"Error initializing: {e}")

PYTHON_EOF

echo ""
echo "================================================================================"
echo "STRACE OUTPUT - Looking for accel access:"
echo "================================================================================"

grep -i "accel" /tmp/strace_tpu.log | head -30

echo ""
echo "================================================================================"
echo "EMULATOR LOG - What did we intercept:"
echo "================================================================================"

grep "accel\|tpu\|sys/class" /tmp/tpu_aggressive.log | head -20

echo ""
echo "================================================================================"
