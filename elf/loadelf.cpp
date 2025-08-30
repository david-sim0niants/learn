#include "loadelf.h"
#include "match_cv.h"

#include <libelf.h>

template<typename T>
auto internal_cast(T* internal)
{
    return reinterpret_cast<MatchCV<T, Elf>*>(internal);
}

void ElfHandle::load(int fd)
{
    internal.handle() = elf_begin(fd, ELF_C_READ, nullptr);
}

void ElfHandle::unload()
{
    elf_end(internal_cast(internal.handle()));
}

ElfView ElfHandle::view() const
{
    return ElfView(internal);
}

ElfView::ElfView(ElfHandle::Internal internal) : internal(internal)
{
}

void* ElfView::find_sym(std::string_view name)
{
    return nullptr;
}
