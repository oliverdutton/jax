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

// TPU v5e characteristics
#define TPU_V5E_CHIP_VERSION 5
#define TPU_V5E_GENERATION "v5e"
#define TPU_V5E_CORES 1
#define TPU_V5E_HBM_SIZE (16ULL * 1024 * 1024 * 1024)  // 16GB

static void init_log(void) {
    if (!logfile) {
        logfile = fopen("/tmp/tpu_emulator.log", "w");
        if (logfile) {
            setvbuf(logfile, NULL, _IONBF, 0);  // Unbuffered
        }
    }
}

static void log_msg(const char *fmt, ...) {
    init_log();
    if (logfile) {
        va_list args;
        va_start(args, fmt);
        vfprintf(logfile, fmt, args);
        va_end(args);
        fflush(logfile);
    }

    // Also to stderr
    va_list args2;
    va_start(args2, fmt);
    vfprintf(stderr, fmt, args2);
    va_end(args2);
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
            log_msg("[EMULATOR] Registered fake fd %d at slot %d\n", fd, i);
            return;
        }
    }
}

static void remove_fake_fd(int fd) {
    for (int i = 0; i < MAX_FAKE_FDS; i++) {
        if (fake_fds[i] == fd) {
            log_msg("[EMULATOR] Removing fake fd %d from slot %d\n", fd, i);
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

// Intercept openat()
int openat(int dirfd, const char *pathname, int flags, ...) {
    static int (*real_openat)(int, const char *, int, ...) = NULL;
    if (!real_openat) {
        real_openat = dlsym(RTLD_NEXT, "openat");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[EMULATOR] openat(\"%s\", flags=0x%x) - returning fake fd\n",
                pathname, flags);
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
        log_msg("[EMULATOR] open(\"%s\", flags=0x%x) - returning fake fd\n",
                pathname, flags);
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
        log_msg("[EMULATOR] access(\"%s\", mode=%d) - SUCCESS\n", pathname, mode);
        return 0;  // Success
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
        log_msg("[EMULATOR] stat(\"%s\")\n", pathname);
        memset(statbuf, 0, sizeof(struct stat));

        if (strstr(pathname, "/sys/")) {
            // Sysfs entries are usually directories or files
            if (strstr(pathname, "accel0")) {
                statbuf->st_mode = S_IFLNK | 0777;  // Symlink
            } else {
                statbuf->st_mode = S_IFDIR | 0755;  // Directory
            }
        } else {
            // /dev/accel* is a character device
            statbuf->st_mode = S_IFCHR | 0666;
            statbuf->st_rdev = makedev(510, 0);
        }
        return 0;
    }

    return real_stat(pathname, statbuf);
}

// Intercept lstat()
int lstat(const char *pathname, struct stat *statbuf) {
    return stat(pathname, statbuf);  // Same as stat for our purposes
}

// Intercept fstat()
int fstat(int fd, struct stat *statbuf) {
    static int (*real_fstat)(int, struct stat *) = NULL;
    if (!real_fstat) {
        real_fstat = dlsym(RTLD_NEXT, "fstat");
    }

    if (is_fake_fd(fd)) {
        log_msg("[EMULATOR] fstat(fd=%d)\n", fd);
        memset(statbuf, 0, sizeof(struct stat));
        // Could be either directory or char device depending on context
        statbuf->st_mode = S_IFCHR | 0666;
        statbuf->st_rdev = makedev(510, 0);
        return 0;
    }

    return real_fstat(fd, statbuf);
}

// Intercept read() - provide TPU version info
ssize_t read(int fd, void *buf, size_t count) {
    static ssize_t (*real_read)(int, void *, size_t) = NULL;
    if (!real_read) {
        real_read = dlsym(RTLD_NEXT, "read");
    }

    if (is_fake_fd(fd)) {
        log_msg("[EMULATOR] read(fd=%d, count=%zu)\n", fd, count);

        // Provide v5e version string
        const char *version_info = "v5e\n";
        size_t len = strlen(version_info);
        if (len > count) len = count;

        memcpy(buf, version_info, len);
        return len;
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
        log_msg("[EMULATOR] write(fd=%d, count=%zu) - pretending success\n", fd, count);
        return count;
    }

    return real_write(fd, buf, count);
}

// Intercept getdents64() for directory listing
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
        log_msg("[EMULATOR] getdents64(fd=%d, count=%zu)\n", fd, count);

        int slot = -1;
        for (int i = 0; i < MAX_FAKE_FDS; i++) {
            if (fake_fds[i] == fd) {
                slot = i;
                break;
            }
        }

        if (slot == -1 || entries_returned[slot] >= 1) {
            if (slot != -1) entries_returned[slot] = 0;
            log_msg("[EMULATOR] getdents64 - EOF\n");
            return 0;  // EOF
        }

        // Return entry for "accel0" symlink
        struct linux_dirent64 *d = (struct linux_dirent64 *)dirp;
        d->d_ino = 1234;
        d->d_off = 1;
        d->d_reclen = sizeof(struct linux_dirent64) + 16;
        d->d_type = DT_LNK;  // Symlink (typical for sysfs)
        strcpy(d->d_name, "accel0");

        entries_returned[slot]++;
        log_msg("[EMULATOR] getdents64 - returned accel0 entry\n");
        return d->d_reclen;
    }

    return real_getdents64(fd, dirp, count);
}

// Intercept readlink() for sysfs symlinks
ssize_t readlink(const char *pathname, char *buf, size_t bufsiz) {
    static ssize_t (*real_readlink)(const char *, char *, size_t) = NULL;
    if (!real_readlink) {
        real_readlink = dlsym(RTLD_NEXT, "readlink");
    }

    if (is_tpu_path(pathname)) {
        log_msg("[EMULATOR] readlink(\"%s\")\n", pathname);

        // Point to a fake device path
        const char *target = "../../devices/pci0000:00/0000:00:04.0/accel0";
        size_t len = strlen(target);
        if (len > bufsiz) len = bufsiz;

        memcpy(buf, target, len);
        return len;
    }

    return real_readlink(pathname, buf, bufsiz);
}

// CRITICAL: Intercept ioctl() with detailed v5e responses
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

        log_msg("[EMULATOR] ioctl(fd=%d, request=0x%lx, argp=%p)\n",
                fd, request, argp);

        if (argp) {
            // Initialize buffer with zeros
            memset(argp, 0, 1024);  // Generous size

            // Try to provide sensible v5e information based on ioctl code
            // Common ioctl patterns:
            // - 0x8xxx = read from device
            // - 0x4xxx = write to device
            // - 0xc xxx = read/write

            // Provide TPU v5e chip information
            uint32_t *u32_buf = (uint32_t *)argp;
            uint64_t *u64_buf = (uint64_t *)argp;
            char *str_buf = (char *)argp;

            // Guess at what libtpu might be asking for
            if ((request & 0xFF00) == 0x8000 || (request & 0xFF00) == 0xC000) {
                // Read ioctl - provide device info
                switch (request & 0xFF) {
                    case 0:  // Maybe chip version
                        u32_buf[0] = TPU_V5E_CHIP_VERSION;
                        log_msg("[EMULATOR]   -> Returned chip version: %d\n", TPU_V5E_CHIP_VERSION);
                        break;
                    case 1:  // Maybe generation string
                        strcpy(str_buf, TPU_V5E_GENERATION);
                        log_msg("[EMULATOR]   -> Returned generation: %s\n", TPU_V5E_GENERATION);
                        break;
                    case 2:  // Maybe core count
                        u32_buf[0] = TPU_V5E_CORES;
                        log_msg("[EMULATOR]   -> Returned cores: %d\n", TPU_V5E_CORES);
                        break;
                    case 3:  // Maybe HBM size
                        u64_buf[0] = TPU_V5E_HBM_SIZE;
                        log_msg("[EMULATOR]   -> Returned HBM size: %llu\n", TPU_V5E_HBM_SIZE);
                        break;
                    default:
                        // Generic success response
                        u32_buf[0] = 0x76356500;  // "v5e\0" in hex
                        log_msg("[EMULATOR]   -> Returned generic v5e marker\n");
                        break;
                }
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
        log_msg("[EMULATOR] close(fd=%d)\n", fd);
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
    log_msg("TPU v5e EMULATOR LOADED\n");
    log_msg("================================================================================\n");
    log_msg("Emulating TPU v5e (Trillium) characteristics:\n");
    log_msg("  - Chip version: %d\n", TPU_V5E_CHIP_VERSION);
    log_msg("  - Generation: %s\n", TPU_V5E_GENERATION);
    log_msg("  - Cores: %d\n", TPU_V5E_CORES);
    log_msg("  - HBM: %llu GB\n", TPU_V5E_HBM_SIZE / (1024*1024*1024));
    log_msg("  - Log file: /tmp/tpu_emulator.log\n");
    log_msg("================================================================================\n");
    log_msg("\n");
}

__attribute__((destructor))
static void cleanup(void) {
    if (logfile) {
        log_msg("\n[EMULATOR] Shutting down\n");
        fclose(logfile);
        logfile = NULL;
    }
}
