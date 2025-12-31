
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
        fprintf(stderr, "[PRELOAD] Intercepted open(\"%s\") - returning fake fd\n", pathname);

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
        fprintf(stderr, "[PRELOAD] Intercepted ioctl(fd=%d, request=0x%lx) - returning success\n",
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
        fprintf(stderr, "[PRELOAD] Intercepted close(fd=%d)\n", fd);
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
        fprintf(stderr, "[PRELOAD] Intercepted read(fd=%d, count=%zu)\n", fd, count);
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
        fprintf(stderr, "[PRELOAD] Intercepted write(fd=%d, count=%zu)\n", fd, count);
        // Pretend we wrote everything
        return count;
    }

    return real_write(fd, buf, count);
}

// Constructor to announce ourselves
__attribute__((constructor))
static void init(void) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "TPU Hardware Detection Bypass Loaded\n");
    fprintf(stderr, "Will intercept /dev/accel* device access\n");
    fprintf(stderr, "========================================\n\n");
}
