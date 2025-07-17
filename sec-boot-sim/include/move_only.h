#pragma once

#include <utility>

template<typename Handle, Handle NULL_HANDLE = Handle()>
class MoveOnly {
public:
    explicit MoveOnly(Handle handle = NULL_HANDLE) noexcept : handle_(handle)
    {
    }

    inline MoveOnly(MoveOnly&& other) noexcept : MoveOnly(std::move(other.handle_))
    {
        other.handle_ = NULL_HANDLE;
    }

    inline MoveOnly& operator=(MoveOnly&& rhs) noexcept
    {
        if (this != &rhs) {
            handle_ = std::move(rhs.handle_);
            rhs.handle_ = NULL_HANDLE;
        }
        return *this;
    }

    inline operator Handle&() noexcept
    {
        return handle_;
    }

    inline operator const Handle&() const noexcept
    {
        return handle_;
    }

    inline Handle& handle() noexcept
    {
        return handle_;
    }

    inline const Handle& handle() const noexcept
    {
        return handle_;
    }

    inline void release() noexcept
    {
        handle_ = NULL_HANDLE;
    }

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

private:
    Handle handle_;
};
