"""
Bypass TPU hardware detection by intercepting device checks.

The key insight: VLIW compilation is STATIC - it doesn't need actual hardware,
just knowledge of the target architecture (v5e). We need to:
1. Make libtpu believe a TPU v5e exists
2. Provide architecture parameters
3. Let the compiler run statically
"""
import os
import sys
import tempfile
import ctypes
from ctypes import CDLL, c_int, c_char_p, POINTER, Structure, c_void_p
import subprocess

print("="*80)
print("BYPASSING TPU HARDWARE DETECTION")
print("="*80)

# Strategy 1: Create fake /dev/accel device
print("\n" + "="*80)
print("STRATEGY 1: Create fake /dev/accel0 device")
print("="*80)

def create_fake_tpu_device():
    """
    Create a fake /dev/accel0 character device that libtpu can detect.
    This requires root but might trick libtpu into initializing.
    """
    print("\nAttempting to create fake /dev/accel0...")

    # Check if we have permission
    if os.geteuid() == 0:
        print("✓ Running as root")
        try:
            # Create character device (major=accel, minor=0)
            # Find the major number for accel devices
            result = subprocess.run(['mknod', '/dev/accel0', 'c', '510', '0'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Created /dev/accel0")
                os.chmod('/dev/accel0', 0o666)
                return True
            else:
                print(f"✗ mknod failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Error creating device: {e}")
    else:
        print("✗ Not running as root - cannot create device nodes")
        print("  (Run with sudo or use LD_PRELOAD approach instead)")

    return False

# Check if device exists
if os.path.exists('/dev/accel0'):
    print("\n✓ /dev/accel0 already exists!")
else:
    print("\n/dev/accel0 does not exist")
    # Try to create it
    if create_fake_tpu_device():
        print("✓ Successfully created fake device")
    else:
        print("✗ Could not create fake device")

# Strategy 2: LD_PRELOAD to intercept open/ioctl calls
print("\n" + "="*80)
print("STRATEGY 2: LD_PRELOAD to intercept hardware detection")
print("="*80)

print("""
Creating a shared library that intercepts:
- open("/dev/accel*") → return fake fd
- ioctl(fd, ...) → return success with fake v5e info
- read/write on fake fd → return appropriate responses
""")

# Create C code for LD_PRELOAD library
preload_code = '''
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <dlfcn.h>
#include <errno.h>

// Track fake TPU device file descriptors
#define MAX_FAKE_FDS 16
static int fake_fds[MAX_FAKE_FDS] = {0};
static int next_fake_fd = 1000; // Start from high number

static int is_fake_fd(int fd) {
    for (int i = 0; i < MAX_FAKE_FDS; i++) {
        if (fake_fds[i] == fd) return 1;
    }
    return 0;
}

static void add_fake_fd(int fd) {
    for (int i = 0; i < MAX_FAKE_FDS; i++) {
        if (fake_fds[i] == 0) {
            fake_fds[i] = fd;
            return;
        }
    }
}

static void remove_fake_fd(int fd) {
    for (int i = 0; i < MAX_FAKE_FDS; i++) {
        if (fake_fds[i] == fd) {
            fake_fds[i] = 0;
            return;
        }
    }
}

// Intercept open()
int open(const char *pathname, int flags, ...) {
    static int (*real_open)(const char *, int, ...) = NULL;
    if (!real_open) {
        real_open = dlsym(RTLD_NEXT, "open");
    }

    // Check if opening a TPU device
    if (pathname && (strstr(pathname, "/dev/accel") || strstr(pathname, "jellyfish"))) {
        fprintf(stderr, "[PRELOAD] Intercepted open(\"%s\") - returning fake fd\\n", pathname);

        // Return a fake fd
        int fake_fd = next_fake_fd++;
        add_fake_fd(fake_fd);

        return fake_fd;
    }

    // Pass through to real open
    return real_open(pathname, flags);
}

// Intercept open64()
int open64(const char *pathname, int flags, ...) {
    return open(pathname, flags);
}

// Intercept ioctl()
int ioctl(int fd, unsigned long request, ...) {
    static int (*real_ioctl)(int, unsigned long, ...) = NULL;
    if (!real_ioctl) {
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] Intercepted ioctl(fd=%d, request=0x%lx) - returning success\\n",
                fd, request);

        // For now, just return success
        // In a real implementation, we'd parse the request and fill appropriate buffers
        return 0;
    }

    // Pass through to real ioctl
    return real_ioctl(fd, request);
}

// Intercept close()
int close(int fd) {
    static int (*real_close)(int) = NULL;
    if (!real_close) {
        real_close = dlsym(RTLD_NEXT, "close");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] Intercepted close(fd=%d)\\n", fd);
        remove_fake_fd(fd);
        return 0;
    }

    return real_close(fd);
}

// Intercept read()
ssize_t read(int fd, void *buf, size_t count) {
    static ssize_t (*real_read)(int, void *, size_t) = NULL;
    if (!real_read) {
        real_read = dlsym(RTLD_NEXT, "read");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] Intercepted read(fd=%d, count=%zu)\\n", fd, count);
        // Return empty read
        return 0;
    }

    return real_read(fd, buf, count);
}

// Intercept write()
ssize_t write(int fd, const void *buf, size_t count) {
    static ssize_t (*real_write)(int, const void *, size_t) = NULL;
    if (!real_write) {
        real_write = dlsym(RTLD_NEXT, "write");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] Intercepted write(fd=%d, count=%zu)\\n", fd, count);
        // Pretend we wrote everything
        return count;
    }

    return real_write(fd, buf, count);
}

// Constructor to announce ourselves
__attribute__((constructor))
static void init(void) {
    fprintf(stderr, "\\n========================================\\n");
    fprintf(stderr, "TPU Hardware Detection Bypass Loaded\\n");
    fprintf(stderr, "Will intercept /dev/accel* device access\\n");
    fprintf(stderr, "========================================\\n\\n");
}
'''

preload_lib_path = "/home/user/jax/tpu_bypass.so"

print(f"\nWriting LD_PRELOAD library code...")
with open("/home/user/jax/tpu_bypass.c", "w") as f:
    f.write(preload_code)
print(f"✓ Wrote tpu_bypass.c")

print(f"\nCompiling LD_PRELOAD library...")
try:
    result = subprocess.run([
        'gcc', '-shared', '-fPIC', '-o', preload_lib_path,
        '/home/user/jax/tpu_bypass.c', '-ldl'
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Compiled to {preload_lib_path}")
    else:
        print(f"✗ Compilation failed:")
        print(result.stderr)
        sys.exit(1)
except Exception as e:
    print(f"✗ Error compiling: {e}")
    sys.exit(1)

# Strategy 3: Set TPU architecture via environment variables
print("\n" + "="*80)
print("STRATEGY 3: Force TPU v5e architecture parameters")
print("="*80)

# Set environment variables that might tell libtpu about the architecture
env_vars = {
    'TPU_CHIPS_PER_HOST_BOUNDS': '1,1,1',
    'TPU_HOST_BOUNDS': '1,1,1',
    'TPU_MESH_CONTROLLER_ADDRESS': 'localhost:8888',
    'TPU_MESH_CONTROLLER_PORT': '8888',
    'TPU_NAME': 'tpu-v5e-fake',
    'TPU_LOAD_LIBRARY': '1',
    'CLOUD_TPU_TASK_ID': '0',
    'TPU_CHIPS': '1',
    'TPU_ACCELERATOR_TYPE': 'v5litepod-1',
    'TPU_WORKER_ID': '0',
    'TPU_WORKER_HOSTNAMES': 'localhost',
}

print("\nSetting TPU architecture environment variables:")
for key, value in env_vars.items():
    os.environ[key] = value
    print(f"  {key}={value}")

# Strategy 4: Test with LD_PRELOAD
print("\n" + "="*80)
print("STRATEGY 4: Test with LD_PRELOAD active")
print("="*80)

print(f"\nTo test, run:")
print(f"  LD_PRELOAD={preload_lib_path} python test_tpu_init.py")
print(f"\nThe preload library will intercept /dev/accel* access and pretend it exists.")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
The LD_PRELOAD library is now compiled. It will:
1. Intercept open("/dev/accel*") and return a fake file descriptor
2. Intercept ioctl() calls on fake fd and return success
3. This should allow libtpu to believe a TPU exists

To proceed:
1. Run with LD_PRELOAD to bypass device detection
2. Set LIBTPU_INIT_ARGS to dump LLO
3. Provide v5e architecture info
4. Let the compiler generate VLIW bundles statically

Let's test it!
""")
