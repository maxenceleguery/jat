#pragma once

#include <cuda_runtime.h>

enum Device {
    CPU,
    CUDA,
};

class CudaReady {
    public:
        virtual __host__ void cuda() = 0;
        virtual __host__ void cpu() = 0;
        virtual __host__ void free() = 0;
        virtual __host__ __device__ Device get_device() const = 0;
};
