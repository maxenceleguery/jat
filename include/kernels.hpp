#pragma once

namespace jat {
    template<typename T>
    class Tensor;
}

namespace kernel {
    void launch_add_kernel(jat::Tensor<float> output, const jat::Tensor<float> input1, const jat::Tensor<float> input2);
}

namespace jat {
    template<typename T>
    Tensor<T> operator+(const Tensor<T>& t1, const Tensor<T>& t2);
} 

