#include <iostream>

#include <fcntl.h>
#include <unistd.h>

#include "loadelf.h"

void usage(char* self)
{
    std::cerr << self << " elf-filename [custom-entry]\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        return EXIT_FAILURE;
    }

    ElfHandle elf;
    elf.load(fd);

    try {
        reinterpret_cast<void (*)()>(elf.view().find_sym("main"))();
    } catch (...) {
        close(fd);
        throw;
    }

    close(fd);
    return EXIT_SUCCESS;
}
