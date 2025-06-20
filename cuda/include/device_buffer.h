#pragma once

#include <cstddef>
#include <stdexcept>

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_(size)
    {
        cudaError_t e = cudaMalloc(&data_, size * sizeof(T));
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }

    explicit DeviceBuffer(const T *host_data, std::size_t size) : DeviceBuffer(size)
    {
        copy_from_host(host_data);
    }

    ~DeviceBuffer()
    {
        cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer& rhs) : DeviceBuffer(rhs.size_)
    {
        copy_from_device(rhs.data_);
    }

    DeviceBuffer(DeviceBuffer&& rhs) noexcept
    {
        swap(rhs);
    }

    DeviceBuffer& operator=(const DeviceBuffer& rhs)
    {
        if (this == &rhs)
            return *this;
        DeviceBuffer copy = rhs;
        swap(copy);
        return *this;
    }

    DeviceBuffer& operator=(DeviceBuffer&& rhs) noexcept
    {
        if (this != &rhs)
            swap(rhs);
        return *this;
    }

    void swap(DeviceBuffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    void copy_from_host(const T *host_data)
    {
        cudaError_t e = cudaMemcpy(data_, host_data, mem_size(), cudaMemcpyHostToDevice);
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }

    void copy_to_host(T *host_data) const
    {
        cudaError_t e = cudaMemcpy(host_data, data_, mem_size(), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }

    void copy_from_device(const T *device_data)
    {
        cudaError_t e = cudaMemcpy(data_, device_data, mem_size(), cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }

    void copy_to_device(T *device_data) const
    {
        cudaError_t e = cudaMemcpy(device_data, data_, mem_size(), cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }

    inline T *data() noexcept
    {
        return data_;
    }

    inline const T *data() const noexcept
    {
        return data_;
    }

    inline std::size_t size() const noexcept
    {
        return size_;
    }

    inline std::size_t mem_size() const noexcept
    {
        return size_ * sizeof(T);
    }

private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
};
