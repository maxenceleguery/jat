#include <gtest/gtest.h>
#include "jat.h"

TEST(ArrayTest, CpuTest) {

    jat::Array<float> array;
    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);

    EXPECT_FLOAT_EQ(array[0], 1.f);
    EXPECT_FLOAT_EQ(array[1], 2.f);
    EXPECT_FLOAT_EQ(array[2], 3.f);

    EXPECT_EQ(array.size(), 3);

    array.fill_(1.f);
    EXPECT_FLOAT_EQ(array[0], 1.f);
    EXPECT_FLOAT_EQ(array[1], 1.f);
    EXPECT_FLOAT_EQ(array[2], 1.f);

    array.free();
}

TEST(ArrayTest2, CpuTest) {

    jat::Array<float> array = jat::Array<float>({1.f, 2.f, 3.f});

    EXPECT_FLOAT_EQ(array[0], 1.f);
    EXPECT_FLOAT_EQ(array[1], 2.f);
    EXPECT_FLOAT_EQ(array[2], 3.f);

    EXPECT_EQ(array.size(), 3);

    array.fill_(1.f);
    EXPECT_FLOAT_EQ(array[0], 1.f);
    EXPECT_FLOAT_EQ(array[1], 1.f);
    EXPECT_FLOAT_EQ(array[2], 1.f);

    array.free();
}

TEST(TensorAllocationTest, CpuTest) {
    jat::Tensor<float> tensor1 = jat::ones<float>({5});
    for (int i = 0; i<5; i++) {
        EXPECT_FLOAT_EQ(tensor1.at({i}), 1.f);
    }

    jat::Tensor<float> tensor2 = jat::zeros<float>({5});
    for (int i = 0; i<5; i++) {
        EXPECT_FLOAT_EQ(tensor2.at({i}), 0.f);
    }

    EXPECT_EQ(tensor1.shape.size(), 1);
    EXPECT_EQ(tensor1.shape[0], 5);

    tensor1.free();
    tensor2.free();
}

TEST(ArrayCopyTest, CpuTest) {
    jat::Array<float> array = jat::Array<float>({1.f, 2.f, 3.f});
    jat::Array<float> array2 = array.copy();

    EXPECT_EQ(array.size(), array2.size());
    for (int i = 0; i<array.size(); i++) {
        EXPECT_FLOAT_EQ(array[i], array2[i]);
    }

    array.free();
    array2.free();
}

TEST(ArrayCopyTest, GpuTest) {
    jat::Array<float> array = jat::Array<float>({1.f, 2.f, 3.f});
    array.cuda();
    jat::Array<float> array2 = array.copy();

    array.cpu();
    array2.cpu();

    EXPECT_EQ(array.size(), array2.size());
    for (int i = 0; i<array.size(); i++) {
        EXPECT_FLOAT_EQ(array[i], array2[i]);
    }

    array.free();
    array2.free();
}

TEST(TensorAddTest1D, CpuTest) {
    jat::Tensor<float> tensor1 = jat::ones<float>({5});
    jat::Tensor<float> tensor2 = jat::ones<float>({5});
    jat::Tensor<float> tensor3 = tensor1 + tensor2;

    for (int i = 0; i<5; i++) {
        EXPECT_FLOAT_EQ(tensor3.at({i}), 2.);
    }
    tensor1.free();
    tensor2.free();
    tensor3.free();
}

TEST(TensorAddTest1D, GpuTest) {
    jat::Tensor<float> tensor1 = jat::ones<float>({5});
    jat::Tensor<float> tensor2 = jat::ones<float>({5});
    tensor1.cuda();
    tensor2.cuda();

    jat::Tensor<float> tensor3 = tensor1 + tensor2;

    tensor1.cpu();
    tensor2.cpu();
    tensor3.cpu();
    for (int i = 0; i<5; i++) {
        EXPECT_FLOAT_EQ(tensor3.at({i}), 2.);
    }
    tensor1.free();
    tensor2.free();
    tensor3.free();
}

TEST(TensorAddTest2D, CpuTest) {
    jat::Tensor<float> tensor1 = jat::ones<float>({5, 3});
    jat::Tensor<float> tensor2 = jat::ones<float>({5, 3});

    jat::Tensor<float> tensor3 = tensor1 + tensor2;

    for (int i = 0; i<5; i++) {
        for (int j = 0; i<3; i++) {
            EXPECT_FLOAT_EQ(tensor3.at({i, j}), 2.);
        }
    }
    tensor1.free();
    tensor2.free();
    tensor3.free();
}

TEST(TensorAddTest2D, GpuTest) {
    jat::Tensor<float> tensor1 = jat::ones<float>({5, 3});
    jat::Tensor<float> tensor2 = jat::ones<float>({5, 3});
    tensor1.cuda();
    tensor2.cuda();

    jat::Tensor<float> tensor3 = tensor1 + tensor2;

    tensor1.cpu();
    tensor2.cpu();
    tensor3.cpu();
    for (int i = 0; i<5; i++) {
        for (int j = 0; i<3; i++) {
            EXPECT_FLOAT_EQ(tensor3.at({i, j}), 2.);
        }
    }
    tensor1.free();
    tensor2.free();
    tensor3.free();
}