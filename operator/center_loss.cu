/*!
 * Copyright (c) 2016 by Contributors
 * \file center_loss.cu
 * \brief center loss from <A Discriminative Feature Learning Approach for Deep Face Recogfnition>
 * \author luoyetx
 */
#include "./center_loss-inl.h"

namespace mshadow {
namespace cuda {

template<typename DType>
__global__ void CenterLossForwardKernel(const int num, const int k,
    const DType *data, const DType *label, const DType *center, DType *diff) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < num;
       index += blockDim.x * gridDim.x) {
    const int y = index / k;
    const int x = index % k;
    const int center_y = static_cast<int>(label[y]);
    const int center_x = x;
    const int center_idx = center_x + center_y * k;
    diff[index] = data[index] - center[center_idx];
  }
}

template<typename DType>
__global__ void CenterLossBackwardKernel(const int m, const int n, const int k,
    const DType *diff, const DType *label, DType *center, DType *workspace, DType alpha) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < m;
       index += blockDim.x * gridDim.x) {
    DType *workspace_ = workspace + index * k;
    DType *center_ = center + index * k;
    for (int l = 0; l < k; ++l) {
      workspace_[l] = 0;
    }
    int count = 0;
    for (int i = 0; i < n; ++i) {
      int j = static_cast<int>(label[i]);
      if (j == index) {
        ++count;
        for (int l = 0; l < k; ++l) {
          workspace_[l] += diff[i * k + l];
        }
      }
    }
    for (int l = 0; l < k; ++l) {
      center_[l] += alpha / static_cast<DType>(1 + count) * workspace_[l];
    }
  }
}

template<typename DType>
inline void CenterLossForward(const Tensor<gpu, 2, DType> &data,
                              const Tensor<gpu, 1, DType> &label,
                              const Tensor<gpu, 2, DType> &center,
                              const Tensor<gpu, 2, DType> &diff) {
  const int k = data.size(1);
  const int num = data.shape_.Size();
  const DType *data_ptr = data.dptr_;
  const DType *label_ptr = label.dptr_;
  const DType *center_ptr = center.dptr_;
  DType *diff_ptr = diff.dptr_;
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((num + kBaseThreadNum - 1) / kBaseThreadNum);
  CenterLossForwardKernel<<<dimGrid, dimBlock>>>(num, k, data_ptr, label_ptr, center_ptr, diff_ptr);
}

template<typename DType>
inline void CenterLossBackward(const Tensor<gpu, 2, DType> &diff,
                               const Tensor<gpu, 1, DType> &label,
                               const Tensor<gpu, 2, DType> &center,
                               const Tensor<gpu, 2, DType> &workspace,
                               DType alpha) {
  const int n = diff.size(0);
  const int k = diff.size(1);
  const int m = center.size(0);
  const DType *diff_ptr = diff.dptr_;
  const DType *label_ptr = label.dptr_;
  DType *center_ptr = center.dptr_;
  DType *workspace_ptr = workspace.dptr_;
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((m + kBaseThreadNum - 1) / kBaseThreadNum);
  CenterLossBackwardKernel<<<dimGrid, dimBlock>>>(m, n, k, diff_ptr, label_ptr, center_ptr, workspace_ptr, alpha);
}

}  // namespace cuda

template<typename DType>
inline void CenterLossForward(const Tensor<gpu, 2, DType> &data,
                              const Tensor<gpu, 1, DType> &label,
                              const Tensor<gpu, 2, DType> &center,
                              const Tensor<gpu, 2, DType> &diff) {
  cuda::CenterLossForward(data, label, center, diff);
}

template<typename DType>
inline void CenterLossBackward(const Tensor<gpu, 2, DType> &diff,
                               const Tensor<gpu, 1, DType> &label,
                               const Tensor<gpu, 2, DType> &center,
                               const Tensor<gpu, 2, DType> &workspace,
                               DType alpha) {
  cuda::CenterLossBackward(diff, label, center, workspace, alpha);
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(CenterLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CenterLossOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet
