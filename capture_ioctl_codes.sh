#!/bin/bash

# First, let's capture what ioctl codes libtpu actually uses
echo "================================================================================"
echo "STEP 1: Capture ioctl codes that libtpu uses"
echo "================================================================================"

export TPU_LOAD_LIBRARY='1'
export JAX_PLATFORMS='cpu,tpu'

# Run with strace to see ioctl codes
timeout 10 strace -e trace=ioctl -f python3 << 'PYTHON_EOF' 2>&1 | grep -E "ioctl.*accel|ioctl.*510" | head -20

import os
os.environ['JAX_PLATFORMS'] = 'cpu,tpu'

try:
    import jax
    devices = jax.devices()
except:
    pass

PYTHON_EOF

echo ""
echo "================================================================================"
