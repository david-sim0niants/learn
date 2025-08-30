#pragma once

#include <string_view>

#include "move_only.h"

class ElfView;

class ElfHandle {
public:
    using Internal = void*;

    ElfHandle() = default;
    ElfHandle(ElfHandle&&) = default;
    ElfHandle& operator=(ElfHandle&&) = default;

    ~ElfHandle()
    {
        unload();
    }

    void load(int fd);
    void unload();
    ElfView view() const;

private:
    MoveOnly<Internal> internal;
};

class ElfView {
    friend class ElfHandle;
public:
    void* find_sym(std::string_view name);

private:
    explicit ElfView(ElfHandle::Internal);

    ElfHandle::Internal internal;
};
