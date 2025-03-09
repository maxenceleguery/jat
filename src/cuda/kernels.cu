#include <cuda_runtime.h>
#include "kernels.hpp"
#include "tensor.hpp"

__global__ void add_kernel(jat::Tensor<float> output, jat::Tensor<float> input1, jat::Tensor<float> input2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output.total_size()) {
        output[idx] = input1[idx] + input2[idx];
    }
}

namespace kernel {
    void launch_add_kernel(jat::Tensor<float> output, jat::Tensor<float> input1, jat::Tensor<float> input2) {
        if (output._is_same_size(input1) && output._is_same_size(input2)) {
            int blockSize = 1024;
            int numBlocks = (output.total_size() + blockSize - 1) / blockSize;
            add_kernel<<<numBlocks, blockSize>>>(output, input1, input2);
            cudaDeviceSynchronize();
        } else {
            throw std::runtime_error("Wrong shape for addition. Got "+input1.shape.to_string()+" and "+input1.shape.to_string()+" with output "+output.shape.to_string());
        }
    }
}
