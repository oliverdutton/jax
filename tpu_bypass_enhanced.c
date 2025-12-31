#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysmacros.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>

// Track fake TPU device file descriptors
#define MAX_FAKE_FDS 16
static int fake_fds[MAX_FAKE_FDS] = {0};
static int next_fake_fd = 1000;

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

static int is_tpu_path(const char *path) {
    if (!path) return 0;
    return (strstr(path, "/dev/accel") != NULL ||
            strstr(path, "jellyfish") != NULL ||
            strstr(path, "/sys/class/accel") != NULL);
}

// Intercept access() - check if file exists
int access(const char *pathname, int mode) {
    static int (*real_access)(const char *, int) = NULL;
    if (!real_access) {
        real_access = dlsym(RTLD_NEXT, "access");
    }

    if (is_tpu_path(pathname)) {
        fprintf(stderr, "[PRELOAD] access(\"%s\", %d) - returning success\n", pathname, mode);
        return 0;  // Pretend it exists and is accessible
    }

    return real_access(pathname, mode);
}

// Intercept stat() - get file info
int stat(const char *pathname, struct stat *statbuf) {
    static int (*real_stat)(const char *, struct stat *) = NULL;
    if (!real_stat) {
        real_stat = dlsym(RTLD_NEXT, "stat");
    }

    if (is_tpu_path(pathname)) {
        fprintf(stderr, "[PRELOAD] stat(\"%s\") - returning fake TPU device info\n", pathname);
        memset(statbuf, 0, sizeof(struct stat));
        statbuf->st_mode = S_IFCHR | 0666;  // Character device, rw-rw-rw-
        statbuf->st_rdev = makedev(510, 0);  // Major 510, Minor 0
        return 0;
    }

    return real_stat(pathname, statbuf);
}

// Intercept lstat()
int lstat(const char *pathname, struct stat *statbuf) {
    static int (*real_lstat)(const char *, struct stat *) = NULL;
    if (!real_lstat) {
        real_lstat = dlsym(RTLD_NEXT, "lstat");
    }

    if (is_tpu_path(pathname)) {
        fprintf(stderr, "[PRELOAD] lstat(\"%s\") - returning fake TPU device info\n", pathname);
        memset(statbuf, 0, sizeof(struct stat));
        statbuf->st_mode = S_IFCHR | 0666;
        statbuf->st_rdev = makedev(510, 0);
        return 0;
    }

    return real_lstat(pathname, statbuf);
}

// Intercept open()
int open(const char *pathname, int flags, ...) {
    static int (*real_open)(const char *, int, ...) = NULL;
    if (!real_open) {
        real_open = dlsym(RTLD_NEXT, "open");
    }

    if (is_tpu_path(pathname)) {
        fprintf(stderr, "[PRELOAD] open(\"%s\") - returning fake fd\n", pathname);
        int fake_fd = next_fake_fd++;
        add_fake_fd(fake_fd);
        return fake_fd;
    }

    return real_open(pathname, flags);
}

// Intercept open64()
int open64(const char *pathname, int flags, ...) {
    return open(pathname, flags);
}

// Intercept ioctl() - this is critical for TPU detection
int ioctl(int fd, unsigned long request, ...) {
    static int (*real_ioctl)(int, unsigned long, ...) = NULL;
    if (!real_ioctl) {
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] ioctl(fd=%d, request=0x%lx) - returning success\n",
                fd, request);

        // Parse varargs to get the third argument (if any)
        va_list args;
        va_start(args, request);
        void *argp = va_arg(args, char *);
        va_end(args);

        // Fill buffer with zeros (safe default for most ioctls)
        if (argp) {
            // We don't know the size, but filling some bytes won't hurt
            // In practice, libtpu will query device info via ioctl
            // We'd need to reverse engineer the exact structs it expects
            memset(argp, 0, 256);  // Guess at reasonable buffer size

            // Try to fill in some "v5e" info if this looks like a device query
            // This is speculative - we'd need to know the exact ioctl commands
            char *buf = (char *)argp;
            if (request == 0x4008 ||  // Common device info ioctl
                request == 0x8008) {
                strcpy(buf, "v5e");  // Pretend we're a v5e
            }
        }

        return 0;  // Success
    }

    return real_ioctl(fd, request);
}

// Intercept close()
int close(int fd) {
    static int (*real_close)(int) = NULL;
    if (!real_close) {
        real_close = dlsym(RTLD_NEXT, "close");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] close(fd=%d)\n", fd);
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
        fprintf(stderr, "[PRELOAD] read(fd=%d, count=%zu) - returning 0 (EOF)\n", fd, count);
        return 0;  // EOF
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
        fprintf(stderr, "[PRELOAD] write(fd=%d, count=%zu) - pretending success\n", fd, count);
        return count;  // Pretend we wrote everything
    }

    return real_write(fd, buf, count);
}

// Intercept readdir() - for when libtpu scans /dev/ for accel devices
struct dirent *readdir(DIR *dirp) {
    static struct dirent *(*real_readdir)(DIR *) = NULL;
    if (!real_readdir) {
        real_readdir = dlsym(RTLD_NEXT, "readdir");
    }

    static struct dirent fake_dirent;
    static int fake_accel_returned = 0;

    // Get real entry
    struct dirent *entry = real_readdir(dirp);

    // If we're scanning /dev/, inject a fake accel0 entry
    // (This is a hack - we'd need to track which DIR* is /dev/)
    if (!fake_accel_returned && entry && entry->d_name[0] == '.') {
        // Inject our fake entry before returning the first real entry
        fake_accel_returned = 1;
        memset(&fake_dirent, 0, sizeof(fake_dirent));
        strcpy(fake_dirent.d_name, "accel0");
        fake_dirent.d_type = DT_CHR;  // Character device
        fprintf(stderr, "[PRELOAD] readdir() - injecting fake accel0 entry\n");
        return &fake_dirent;
    }

    return entry;
}

// Constructor
__attribute__((constructor))
static void init(void) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Enhanced TPU Bypass Loaded\n");
    fprintf(stderr, "Intercepting:\n");
    fprintf(stderr, "  - open/stat/access on /dev/accel*\n");
    fprintf(stderr, "  - ioctl for device queries\n");
    fprintf(stderr, "  - readdir for /dev/ scanning\n");
    fprintf(stderr, "========================================\n\n");
}
