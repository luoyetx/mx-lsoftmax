/*!
 * Copyright (c) 2016 by Contributors
 * \file center_loss.cc
 * \brief center loss from <A Discriminative Feature Learning Approach for Deep Face Recogfnition>
 * \author luoyetx
 */
#include "./center_loss-inl.h"

namespace mshadow {

template<typename DType>
inline void CenterLossForward(const Tensor<cpu, 2, DType> &data,
                              const Tensor<cpu, 1, DType> &label,
                              const Tensor<cpu, 2, DType> &center,
                              const Tensor<cpu, 2, DType> &diff) {
  const int n = data.size(0);
  for (int i = 0; i < n; ++i) {
    diff[i] = data[i] - center[static_cast<int>(label[i])];
  }
}

template<typename DType>
inline void CenterLossBackward(const Tensor<cpu, 2, DType> &diff,
                               const Tensor<cpu, 1, DType> &label,
                               const Tensor<cpu, 2, DType> &center,
                               const Tensor<cpu, 2, DType> &workspace,
                               DType alpha) {
  using namespace mshadow::expr;
  std::map<int, std::vector<int> > label_cnt;
  const int n = diff.size(0);
  for (int i = 0; i < n; ++i) {
    auto& item = label_cnt[static_cast<int>(label[i])];
    item.push_back(i);
  }
  for (auto& kv : label_cnt) {
    auto i = kv.first;
    auto& idx = kv.second;
    workspace[0] = DType(0);
    for (auto j : idx) {
      workspace[0] += diff[j];
    }
    center[i] += workspace[0]*ScalarExp<DType>(alpha / (1 + idx.size()));
  }
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(CenterLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CenterLossOp<cpu, DType>(param);
  })
  return op;
}

Operator *CenterLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(CenterLossParam);

MXNET_REGISTER_OP_PROPERTY(CenterLoss, CenterLossProp)
.describe("Center loss")
.add_argument("data", "Symbol", "data")
.add_argument("label", "Symbol", "label")
.add_arguments(CenterLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
