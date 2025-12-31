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

#define MAX_FAKE_FDS 16
static int fake_fds[MAX_FAKE_FDS] = {0};
static int next_fake_fd = 1000;

static int entries_returned[MAX_FAKE_FDS] = {0};

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
            strstr(path, "jellyfish") != NULL);
}

// Intercept openat()
int openat(int dirfd, const char *pathname, int flags, ...) {
    static int (*real_openat)(int, const char *, int, ...) = NULL;
    if (!real_openat) {
        real_openat = dlsym(RTLD_NEXT, "openat");
    }

    if (is_tpu_path(pathname)) {
        fprintf(stderr, "[PRELOAD] openat(\"%s\") - returning fake fd\n", pathname);
        int fake_fd = next_fake_fd++;
        add_fake_fd(fake_fd);
        return fake_fd;
    }

    return real_openat(dirfd, pathname, flags);
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

// Intercept access()
int access(const char *pathname, int mode) {
    static int (*real_access)(const char *, int) = NULL;
    if (!real_access) {
        real_access = dlsym(RTLD_NEXT, "access");
    }

    if (is_tpu_path(pathname)) {
        fprintf(stderr, "[PRELOAD] access(\"%s\") - success\n", pathname);
        return 0;
    }

    return real_access(pathname, mode);
}

// Intercept stat()
int stat(const char *pathname, struct stat *statbuf) {
    static int (*real_stat)(const char *, struct stat *) = NULL;
    if (!real_stat) {
        real_stat = dlsym(RTLD_NEXT, "stat");
    }

    if (is_tpu_path(pathname)) {
        fprintf(stderr, "[PRELOAD] stat(\"%s\")\n", pathname);
        memset(statbuf, 0, sizeof(struct stat));
        if (strstr(pathname, "/sys/class/accel")) {
            statbuf->st_mode = S_IFDIR | 0755;
        } else {
            statbuf->st_mode = S_IFCHR | 0666;
            statbuf->st_rdev = makedev(510, 0);
        }
        return 0;
    }

    return real_stat(pathname, statbuf);
}

// Intercept fstat()
int fstat(int fd, struct stat *statbuf) {
    static int (*real_fstat)(int, struct stat *) = NULL;
    if (!real_fstat) {
        real_fstat = dlsym(RTLD_NEXT, "fstat");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] fstat(fd=%d)\n", fd);
        memset(statbuf, 0, sizeof(struct stat));
        statbuf->st_mode = S_IFDIR | 0755;
        return 0;
    }

    return real_fstat(fd, statbuf);
}

// Intercept getdents64()
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
        fprintf(stderr, "[PRELOAD] getdents64(fd=%d)\n", fd);

        // Find which fake_fd slot this is
        int slot = -1;
        for (int i = 0; i < MAX_FAKE_FDS; i++) {
            if (fake_fds[i] == fd) {
                slot = i;
                break;
            }
        }

        if (slot == -1 || entries_returned[slot] >= 1) {
            if (slot != -1) entries_returned[slot] = 0;
            return 0;  // EOF
        }

        // Return fake entry for "accel0"
        struct linux_dirent64 *d = (struct linux_dirent64 *)dirp;
        d->d_ino = 3;
        d->d_off = 1;
        d->d_reclen = sizeof(struct linux_dirent64) + 8;
        d->d_type = DT_LNK;
        strcpy(d->d_name, "accel0");

        entries_returned[slot]++;
        return d->d_reclen;
    }

    return real_getdents64(fd, dirp, count);
}

// Intercept ioctl()
int ioctl(int fd, unsigned long request, ...) {
    static int (*real_ioctl)(int, unsigned long, ...) = NULL;
    if (!real_ioctl) {
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    }

    if (is_fake_fd(fd)) {
        fprintf(stderr, "[PRELOAD] ioctl(fd=%d, 0x%lx)\n", fd, request);

        va_list args;
        va_start(args, request);
        void *argp = va_arg(args, char *);
        va_end(args);

        if (argp) {
            memset(argp, 0, 256);
        }

        return 0;
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

__attribute__((constructor))
static void init(void) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "TPU SysFS Bypass Loaded - v3\n");
    fprintf(stderr, "Faking /sys/class/accel/ directory\n");
    fprintf(stderr, "========================================\n\n");
}
