#define _GNU_SOURCE
#include <fcntl.h>
#include <liburing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QUEUE_DEPTH 1
#define READ_SIZE (64 << 10)

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Provide a filename\n");
        return EXIT_FAILURE;
    }

    const char *fn = argv[1];

    int fd = open(fn, O_RDONLY);
    if (fd < 0) {
        perror("open()");
        return EXIT_FAILURE;
    }

    char *buf;
    if ((errno = posix_memalign((void **)&buf, 4096, READ_SIZE)) != 0) {
        perror("posix_memalign()");
        return EXIT_FAILURE;
    }
    memset(buf, 0, READ_SIZE);

    struct io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0) {
        perror("io_uring_queue_init()");
        return EXIT_FAILURE;
    }

    off_t off = 0;
    while (true) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buf, READ_SIZE, off);
        io_uring_submit(&ring);

        struct io_uring_cqe *cqe;
        io_uring_wait_cqe(&ring, &cqe);

        if (cqe->res <= 0) {
            if (cqe->res < 0)
                fprintf(stderr, "Async read failed: %s\n", strerror(-cqe->res));
            break;
        }

        fwrite(buf, 1, cqe->res, stdout);
        off += cqe->res;

        io_uring_cqe_seen(&ring, cqe);
    }

    fflush(stdout);

    io_uring_queue_exit(&ring);
    close(fd);
    free(buf);

    return EXIT_SUCCESS;
}
