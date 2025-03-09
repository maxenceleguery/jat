#pragma once

#include "array.hpp"
#include "kernels.hpp"

namespace jat {

    typedef Array<size_t> Shape;

    template<typename T>
    class Tensor : public CudaReady {
        private:
            Array<T> data;
        public:
            Shape shape;

            Tensor() = delete;
            Tensor(const Array<T> array) {
                data = array;
                shape = Shape({array.size()});
                if (array.get_device() == Device::CUDA) {
                    shape.cuda();
                }
            };
            Tensor(const Array<T> array, const Shape shape) : data(array), shape(shape) {};
            ~Tensor() {};
            
            __host__ __device__
            T at(const std::initializer_list<size_t> indexes) const {
                size_t offset = 0;
                size_t multiplier = 1;
                auto it = indexes.end();
                for (int i = shape.size() - 1; i >= 0; --i) {
                    --it;
                    offset += (*it) * multiplier;
                    multiplier *= shape[i];
                }
                return data[offset];
            }

            template<typename I>
            __host__ __device__
            T operator[](const I i) const {
                return data[i];
            }

            template<typename I>
            __host__ __device__
            T& operator[](const I i) {            
                return data[i];
            }

            __host__
            size_t size(const size_t dim) const {
                if (dim >= shape.size()) {
                    throw std::runtime_error("Tensor has only "+std::to_string(shape.size())+" dimensions. Got "+std::to_string(dim));
                }
                return shape[dim];
            }

            __host__
            Tensor<T> copy() const {
                Tensor<T> tensor = Tensor<T>(data.copy(), shape.copy());
                return tensor;
            }
            
            template<typename U>
            __host__ __device__
            bool _is_same_size(const Tensor<U>& other) const {
                if (shape.size() != other.shape.size()) return false;

                for (int i = 0; i<shape.size(); i++) {
                    if (shape[i] != other.shape[i]) {
                        return false;
                    }
                }
                return true;
            }
            
            __host__ __device__
            size_t total_size() const {
                return data.size();
            }

            __host__
            void cuda() override {
                data.cuda();
                shape.cuda();
            }

            __host__
            void cpu() override {
                data.cpu();
                shape.cpu();
            }

            __host__
            void free() override {
                data.free();
                shape.free();
            }

            __host__ __device__
            Device get_device() const override {
                return data.get_device();
            }

            __host__ __device__
            friend Tensor<T> operator+<>(const Tensor<T>& t1, const Tensor<T>& t2);
    };

    template<typename T>
    Tensor<T> zeros(const std::initializer_list<size_t> shape) {
        size_t total_size = 1;
        for (size_t i : shape) {
            total_size *= i;
        }
        Array<T> array(total_size);
        array.fill_(0);
        return Tensor<T>(array);
    }

    template<typename T>
    Tensor<T> ones(const std::initializer_list<size_t> shape) {
        size_t total_size = 1;
        for (size_t i : shape) {
            total_size *= i;
        }
        Array<T> array(total_size);
        array.fill_(1);
        return Tensor<T>(array);
    }

    template<typename T>
    Tensor<T> operator+(const Tensor<T>& t1, const Tensor<T>& t2) {
        if (!t1._is_same_size(t2)) {
            throw std::runtime_error("Tensors must have the same size for addition.");
        }

        if (t1.get_device() != t1.get_device()) {
            throw std::runtime_error("Tensors must be on same device.");
        }

        Tensor<T> result = t1.copy();

        if (t1.get_device() == Device::CUDA) {
            kernel::launch_add_kernel(result, t1, t2);
        } else {
            for (size_t i = 0; i < result.total_size(); i++) {
                result[i] = t1[i] + t2[i];
            }
        }
        return result;
    }

}
