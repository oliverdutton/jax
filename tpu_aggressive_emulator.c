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
#include <sys/syscall.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>

#define MAX_FAKE_FDS 32
static int fake_fds[MAX_FAKE_FDS] = {0};
static int next_fake_fd = 1000;
static int entries_returned[MAX_FAKE_FDS] = {0};

static FILE *logfile = NULL;

static void init_log(void) {
    if (!logfile) {
        logfile = fopen("/tmp/tpu_aggressive.log", "w");
        if (logfile) setvbuf(logfile, NULL, _IONBF, 0);
    }
}

static void log_msg(const char *fmt, ...) {
    init_log();
    va_list args;
    va_start(args, fmt);
    if (logfile) vfprintf(logfile, fmt, args);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

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
            entries_returned[i] = 0;
            log_msg("[AGGRESSIVE] Added fake fd %d\n", fd);
            return;
        }
    }
}

static void remove_fake_fd(int fd) {
    for (int i = 0; i < MAX_FAKE_FDS; i++) {
        if (fake_fds[i] == fd) {
            fake_fds[i] = 0;
            entries_returned[i] = 0;
            return;
        }
    }
}

static int is_tpu_path(const char *path) {
    if (!path) return 0;
    return (strstr(path, "/dev/accel") != NULL ||
            strstr(path, "/sys/class/accel") != NULL ||
            strstr(path, "/sys/devices") != NULL && strstr(path, "accel") != NULL);
}

// Intercept ALL stat variants
int __xstat(int ver, const char *pathname, struct stat *statbuf) {
    static int (*real_xstat)(int, const char *, struct stat *) = NULL;
    if (!real_xstat) {
        real_xstat = dlsym(RTLD_NEXT, "__xstat");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] __xstat(\"%s\")\n", pathname);
        memset(statbuf, 0, sizeof(struct stat));
        if (strstr(pathname, "/sys/")) {
            statbuf->st_mode = S_IFDIR | 0755;
        } else {
            statbuf->st_mode = S_IFCHR | 0666;
            statbuf->st_rdev = makedev(510, 0);
        }
        return 0;
    }

    return real_xstat(ver, pathname, statbuf);
}

int __lxstat(int ver, const char *pathname, struct stat *statbuf) {
    static int (*real_lxstat)(int, const char *, struct stat *) = NULL;
    if (!real_lxstat) {
        real_lxstat = dlsym(RTLD_NEXT, "__lxstat");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] __lxstat(\"%s\")\n", pathname);
        memset(statbuf, 0, sizeof(struct stat));
        if (strstr(pathname, "/sys/")) {
            statbuf->st_mode = S_IFDIR | 0755;
        } else {
            statbuf->st_mode = S_IFCHR | 0666;
            statbuf->st_rdev = makedev(510, 0);
        }
        return 0;
    }

    return real_lxstat(ver, pathname, statbuf);
}

int __fxstat(int ver, int fd, struct stat *statbuf) {
    static int (*real_fxstat)(int, int, struct stat *) = NULL;
    if (!real_fxstat) {
        real_fxstat = dlsym(RTLD_NEXT, "__fxstat");
    }

    if (is_fake_fd(fd)) {
        log_msg("[AGGRESSIVE] __fxstat(fd=%d)\n", fd);
        memset(statbuf, 0, sizeof(struct stat));
        statbuf->st_mode = S_IFCHR | 0666;
        statbuf->st_rdev = makedev(510, 0);
        return 0;
    }

    return real_fxstat(ver, fd, statbuf);
}

// Regular stat functions
int stat(const char *pathname, struct stat *statbuf) {
    return __xstat(1, pathname, statbuf);
}

int lstat(const char *pathname, struct stat *statbuf) {
    return __lxstat(1, pathname, statbuf);
}

int fstat(int fd, struct stat *statbuf) {
    return __fxstat(1, fd, statbuf);
}

// Intercept stat64 variants
int __xstat64(int ver, const char *pathname, struct stat64 *statbuf) {
    static int (*real_xstat64)(int, const char *, struct stat64 *) = NULL;
    if (!real_xstat64) {
        real_xstat64 = dlsym(RTLD_NEXT, "__xstat64");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] __xstat64(\"%s\")\n", pathname);
        memset(statbuf, 0, sizeof(struct stat64));
        if (strstr(pathname, "/sys/")) {
            statbuf->st_mode = S_IFDIR | 0755;
        } else {
            statbuf->st_mode = S_IFCHR | 0666;
            statbuf->st_rdev = makedev(510, 0);
        }
        return 0;
    }

    return real_xstat64(ver, pathname, statbuf);
}

// Intercept openat
int openat(int dirfd, const char *pathname, int flags, ...) {
    static int (*real_openat)(int, const char *, int, ...) = NULL;
    if (!real_openat) {
        real_openat = dlsym(RTLD_NEXT, "openat");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] openat(\"%s\", flags=0x%x) -> FAKE FD\n", pathname, flags);
        int fake_fd = next_fake_fd++;
        add_fake_fd(fake_fd);
        return fake_fd;
    }

    return real_openat(dirfd, pathname, flags);
}

// Intercept openat64
int openat64(int dirfd, const char *pathname, int flags, ...) {
    return openat(dirfd, pathname, flags);
}

// Intercept open
int open(const char *pathname, int flags, ...) {
    static int (*real_open)(const char *, int, ...) = NULL;
    if (!real_open) {
        real_open = dlsym(RTLD_NEXT, "open");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] open(\"%s\", flags=0x%x) -> FAKE FD\n", pathname, flags);
        int fake_fd = next_fake_fd++;
        add_fake_fd(fake_fd);
        return fake_fd;
    }

    return real_open(pathname, flags);
}

// Intercept open64
int open64(const char *pathname, int flags, ...) {
    return open(pathname, flags);
}

// Intercept access
int access(const char *pathname, int mode) {
    static int (*real_access)(const char *, int) = NULL;
    if (!real_access) {
        real_access = dlsym(RTLD_NEXT, "access");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] access(\"%s\", mode=%d) -> SUCCESS\n", pathname, mode);
        return 0;
    }

    return real_access(pathname, mode);
}

// Intercept faccessat
int faccessat(int dirfd, const char *pathname, int mode, int flags) {
    static int (*real_faccessat)(int, const char *, int, int) = NULL;
    if (!real_faccessat) {
        real_faccessat = dlsym(RTLD_NEXT, "faccessat");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] faccessat(\"%s\", mode=%d) -> SUCCESS\n", pathname, mode);
        return 0;
    }

    return real_faccessat(dirfd, pathname, mode, flags);
}

// Intercept readlink
ssize_t readlink(const char *pathname, char *buf, size_t bufsiz) {
    static ssize_t (*real_readlink)(const char *, char *, size_t) = NULL;
    if (!real_readlink) {
        real_readlink = dlsym(RTLD_NEXT, "readlink");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[AGGRESSIVE] readlink(\"%s\")\n", pathname);
        const char *target = "../../devices/pci0000:00/0000:00:04.0/accel/accel0";
        size_t len = strlen(target);
        if (len > bufsiz) len = bufsiz;
        memcpy(buf, target, len);
        return len;
    }

    return real_readlink(pathname, buf, bufsiz);
}

// Intercept getdents
struct linux_dirent {
    unsigned long d_ino;
    unsigned long d_off;
    unsigned short d_reclen;
    char d_name[];
};

long getdents(unsigned int fd, struct linux_dirent *dirp, unsigned int count) {
    static long (*real_getdents)(unsigned int, struct linux_dirent *, unsigned int) = NULL;
    if (!real_getdents) {
        real_getdents = dlsym(RTLD_NEXT, "getdents");
    }

    if (is_fake_fd(fd)) {
        log_msg("[AGGRESSIVE] getdents(fd=%d)\n", fd);

        int slot = -1;
        for (int i = 0; i < MAX_FAKE_FDS; i++) {
            if (fake_fds[i] == fd) {
                slot = i;
                break;
            }
        }

        if (slot == -1 || entries_returned[slot] >= 1) {
            if (slot != -1) entries_returned[slot] = 0;
            return 0;
        }

        struct linux_dirent *d = dirp;
        d->d_ino = 1234;
        d->d_off = 1;
        d->d_reclen = sizeof(struct linux_dirent) + 8;
        strcpy(d->d_name, "accel0");

        entries_returned[slot]++;
        return d->d_reclen;
    }

    return real_getdents(fd, dirp, count);
}

// Intercept getdents64
struct linux_dirent64 {
    uint64_t d_ino;
    int64_t d_off;
    unsigned short d_reclen;
    unsigned char d_type;
    char d_name[];
};

ssize_t getdents64(int fd, void *dirp, size_t count) {
    static ssize_t (*real_getdents64)(int, void *, size_t) = NULL;
    if (!real_getdents64) {
        real_getdents64 = dlsym(RTLD_NEXT, "getdents64");
    }

    if (is_fake_fd(fd)) {
        log_msg("[AGGRESSIVE] getdents64(fd=%d)\n", fd);

        int slot = -1;
        for (int i = 0; i < MAX_FAKE_FDS; i++) {
            if (fake_fds[i] == fd) {
                slot = i;
                break;
            }
        }

        if (slot == -1 || entries_returned[slot] >= 1) {
            if (slot != -1) entries_returned[slot] = 0;
            return 0;
        }

        struct linux_dirent64 *d = (struct linux_dirent64 *)dirp;
        d->d_ino = 1234;
        d->d_off = 1;
        d->d_reclen = sizeof(struct linux_dirent64) + 8;
        d->d_type = DT_LNK;
        strcpy(d->d_name, "accel0");

        entries_returned[slot]++;
        return d->d_reclen;
    }

    return real_getdents64(fd, dirp, count);
}

// Intercept ioctl
int ioctl(int fd, unsigned long request, ...) {
    static int (*real_ioctl)(int, unsigned long, ...) = NULL;
    if (!real_ioctl) {
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    }

    if (is_fake_fd(fd)) {
        va_list args;
        va_start(args, request);
        void *argp = va_arg(args, char *);
        va_end(args);

        log_msg("[AGGRESSIVE] ioctl(fd=%d, request=0x%lx)\n", fd, request);

        if (argp) {
            memset(argp, 0, 4096);
            // Try to provide v5e info
            uint32_t *u32 = (uint32_t *)argp;
            u32[0] = 5;  // v5e chip version
            strcpy((char *)argp, "v5e");
        }

        return 0;
    }

    return real_ioctl(fd, request);
}

// Intercept read
ssize_t read(int fd, void *buf, size_t count) {
    static ssize_t (*real_read)(int, void *, size_t) = NULL;
    if (!real_read) {
        real_read = dlsym(RTLD_NEXT, "read");
    }

    if (is_fake_fd(fd)) {
        log_msg("[AGGRESSIVE] read(fd=%d, count=%zu)\n", fd, count);
        const char *data = "v5e\n";
        size_t len = strlen(data);
        if (len > count) len = count;
        memcpy(buf, data, len);
        return len;
    }

    return real_read(fd, buf, count);
}

// Intercept write
ssize_t write(int fd, const void *buf, size_t count) {
    static ssize_t (*real_write)(int, const void *, size_t) = NULL;
    if (!real_write) {
        real_write = dlsym(RTLD_NEXT, "write");
    }

    if (is_fake_fd(fd)) {
        log_msg("[AGGRESSIVE] write(fd=%d, count=%zu)\n", fd, count);
        return count;
    }

    return real_write(fd, buf, count);
}

// Intercept close
int close(int fd) {
    static int (*real_close)(int) = NULL;
    if (!real_close) {
        real_close = dlsym(RTLD_NEXT, "close");
    }

    if (is_fake_fd(fd)) {
        log_msg("[AGGRESSIVE] close(fd=%d)\n", fd);
        remove_fake_fd(fd);
        return 0;
    }

    return real_close(fd);
}

__attribute__((constructor))
static void init(void) {
    init_log();
    log_msg("\n");
    log_msg("================================================================================\n");
    log_msg("AGGRESSIVE TPU v5e EMULATOR LOADED\n");
    log_msg("================================================================================\n");
    log_msg("Intercepting ALL variants:\n");
    log_msg("  - stat/__xstat/__lxstat/__fxstat + 64-bit versions\n");
    log_msg("  - open/openat/open64/openat64\n");
    log_msg("  - access/faccessat\n");
    log_msg("  - getdents/getdents64\n");
    log_msg("  - ioctl/read/write/close\n");
    log_msg("  - readlink\n");
    log_msg("Log file: /tmp/tpu_aggressive.log\n");
    log_msg("================================================================================\n");
    log_msg("\n");
}

__attribute__((destructor))
static void cleanup(void) {
    if (logfile) {
        log_msg("\n[AGGRESSIVE] Shutting down\n");
        fclose(logfile);
    }
}
