#!/bin/bash

# Use strace to see what system calls libtpu makes during initialization

echo "================================================================================"
echo "TRACING LIBTPU SYSTEM CALLS"
echo "================================================================================"

# Set TPU env
export TPU_LOAD_LIBRARY='1'
export TPU_CHIPS='1'
export JAX_PLATFORMS='cpu,tpu'

echo ""
echo "Running with strace to see what libtpu looks for..."
echo ""

# Run with strace, filtering for relevant calls
strace -e trace=open,openat,access,stat,ioctl -f python3 << 'PYTHON_EOF' 2>&1 | grep -E "accel|tpu|jellyfish" | head -50

import os
os.environ['JAX_PLATFORMS'] = 'cpu,tpu'

import jax

try:
    devices = jax.devices()
    print(f"Devices: {devices}")
except Exception as e:
    print(f"Error: {e}")

PYTHON_EOF

echo ""
echo "================================================================================"
