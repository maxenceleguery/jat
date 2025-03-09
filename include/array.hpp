#pragma once

#include <type_traits>
#include <string>
#include <cstring>
#include <iostream>
#include "cuda_ready.hpp"

#ifdef __CUDA_ARCH__
#define DATA data_gpu
#else
#define DATA data_cpu
#endif

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

inline void cudaAssert(const cudaError err, const char *file, const int line) {
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}

static std::string human_rep(const int num_bytes) {
    if (num_bytes < 1000) return std::to_string(num_bytes) + " B (" + std::to_string(num_bytes) + " B)";
    if (num_bytes < 1000 * 1000) return std::to_string((float)num_bytes/1000).substr(0, 5) + " KB (" + std::to_string((float)num_bytes/1024).substr(0, 5) + " KiB)";
    if (num_bytes < 1000 * 1000 * 1000) return std::to_string((float)num_bytes/(1000*1000)).substr(0, 5) + " MB (" + std::to_string((float)num_bytes/(1024*1024)).substr(0, 5) + " MiB)";
    return std::to_string((float)num_bytes/(1000*1000*1000)).substr(0, 5) + " GB (" + std::to_string((float)num_bytes/(1024*1024*1024)).substr(0, 5) + " GiB)";;
}

namespace jat {

template<typename T>
class Array : public CudaReady {
    private:
        T* data_cpu;
        T* data_gpu = nullptr;
        size_t data_size;
        Device device = CPU;

    protected:
        size_t space_used = 0;
        
    public:
        __host__ __device__
        Array() : data_cpu(nullptr), data_size(0) {};

        __host__
        Array(const size_t data_size) : data_size(data_size) {
            data_cpu = new T[data_size];
        };
        
        __host__
        Array(const std::initializer_list<T> items) : Array(items.size()) {
            for (T item : items) {
                push_back(item);
            }
        };

        __host__
        size_t push_back(const T item) {
            if (space_used == data_size) {
                data_size++;
                T* tri_tmp = new T[data_size];
                //std::memcpy(tri_tmp, data_cpu, space_used*sizeof(T));
                
                for (size_t i = 0; i < space_used; i++) {
                    tri_tmp[i] = data_cpu[i];
                }
                if (data_cpu != nullptr)
                    delete[] data_cpu;
                data_cpu = tri_tmp;
            }
            data_cpu[space_used++] = item;
            return space_used-1;
        }

        __host__ __device__
        void fill_(const T value) {
            for (size_t i = 0; i<data_size; i++) {
                DATA[i] = static_cast<T>(value);
            }
            space_used = data_size;
        }

        __host__ __device__
        size_t size() const {
            return space_used;
        }

        __host__
        void clear() {
            space_used = 0;
        }

        __host__
        Array<T> copy() const {
            Array<T> array = Array<T>(data_size);

            if (device == Device::CUDA) {
                array.cuda();
                cudaErrorCheck(cudaMemcpy(array.data_gpu, data_gpu, data_size*sizeof(T), cudaMemcpyDeviceToDevice));
                cudaErrorCheck(cudaMemcpy(array.data_cpu, data_gpu, data_size*sizeof(T), cudaMemcpyDeviceToHost));
            } else {
                std::memcpy(array.data_cpu, data_cpu, data_size*sizeof(T));
            }
            array.space_used = space_used;
            return array;
        }

        template<typename I>
        __host__ __device__
        T operator[](const I i) const {
            if constexpr (std::is_signed<I>::value) {
                if (i < 0)
                    return DATA[(int)space_used + i];
            }
            return DATA[i];
        }

        template<typename I>
        __host__ __device__
        T& operator[](const I i) {            
            if constexpr (std::is_signed<I>::value) {
                if (i < 0)
                    return DATA[(int)space_used + i];
            }
            return DATA[i];
        }
        
        __host__
        void cuda() override {
            if (data_size == 0) return;

            if constexpr (std::is_base_of<CudaReady, T>::value) {
                // Maybe size() must be replace by data_size to handle reserved but no set memory
                for (size_t i=0; i<size(); i++) {
                    data_cpu[i].cuda();
                }
            }
            if (data_gpu == nullptr) {
                //std::cout << "Allocating : " << human_rep(data_size*sizeof(T));
                //allocated_cuda_memory += data_size*sizeof(T);
                cudaErrorCheck(cudaMalloc(&data_gpu, data_size*sizeof(T)));
                //std::cout << " Done" << std::endl;
            }
            //std::cout << "Copying to device.";
            cudaErrorCheck(cudaMemcpy(data_gpu, data_cpu, data_size*sizeof(T), cudaMemcpyHostToDevice));
            //std::cout << " Done" << std::endl;
            device = Device::CUDA;
        }

        __host__
        void cpu() override {
            if (data_size == 0) return;

            if (data_gpu != nullptr) {
                cudaErrorCheck(cudaMemcpy(data_cpu, data_gpu, data_size*sizeof(T), cudaMemcpyDeviceToHost));
            }
            if constexpr (std::is_base_of<CudaReady, T>::value) {
                for (size_t i=0; i<size(); i++) {
                    data_cpu[i].cpu();
                }
            }
            device = Device::CPU;
        }

        __host__
        void free() override {
            if (data_size == 0) return;
            
            if constexpr (std::is_base_of<CudaReady, T>::value) {
                for (size_t i=0; i<size(); i++) {
                    data_cpu[i].free();
                }
            }
            if (data_cpu != nullptr) {
                delete[] data_cpu;
                data_cpu = nullptr;
            }
            if (data_gpu != nullptr) {
                cudaErrorCheck(cudaFree(data_gpu));
                //allocated_cuda_memory -= data_size*sizeof(T);
                data_gpu = nullptr;
            }
        }

        __host__ __device__
        Device get_device() const override {
            return device;
        }

        __host__ std::string to_string() const {
            std::string result = "[";
            for (size_t i = 0; i < space_used; ++i) {
                result += std::to_string(DATA[i]);
                if (i < space_used - 1) {
                    result += ", ";
                }
            }
            result += "]";
            return result;
        }
};

}
