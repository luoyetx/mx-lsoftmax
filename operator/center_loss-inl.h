/*!
 * Copyright (c) 2016 by Contributors
 * \file center_loss-inl.h
 * \brief center loss from <A Discriminative Feature Learning Approach for Deep Face Recogfnition>
 * \author luoyetx
 */
#ifndef MXNET_OPERATOR_CENTER_LOSS_INL_H_
#define MXNET_OPERATOR_CENTER_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace center_loss_enum {
enum CenterLossOpInputs {kData, kLabel};
enum CenterLossOpOutputs {kOut};
enum CenterLossOpAuxiliary {kCenter};
enum CenterLossBackResource {kTempSpace};
}

struct CenterLossParam : public dmlc::Parameter<CenterLossParam> {
  float alpha;
  float scale;
  int num_classes;
  DMLC_DECLARE_PARAMETER(CenterLossParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(0.5)
    .describe("alpha for center loss");
    DMLC_DECLARE_FIELD(scale).set_default(1)
    .describe("grad scale, same as lambda for center loss");
    DMLC_DECLARE_FIELD(num_classes)
    .describe("number of classes");
  }
};

template<typename xpu, typename DType>
class CenterLossOp : public Operator {
 public:
  explicit CenterLossOp(CenterLossParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "CenterLossOp Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "CenterLossOp Output: [output]";
    CHECK_EQ(aux_args.size(), 1) << "CenterLossOp Aux: [center]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[center_loss_enum::kData].size(0);
    Tensor<xpu, 2, DType> data = in_data[center_loss_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[center_loss_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> diff = out_data[center_loss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> center = aux_args[center_loss_enum::kCenter].FlatTo2D<xpu, DType>(s);
    CenterLossForward(data, label, center, diff);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(aux_args.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = out_data[center_loss_enum::kOut].size(0);
    const int k = out_data[center_loss_enum::kOut].size(1);
    Tensor<xpu, 2, DType> diff = out_data[center_loss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[center_loss_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> center = aux_args[center_loss_enum::kCenter].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = in_grad[center_loss_enum::kData].FlatTo2D<xpu, DType>(s);
    // gradient w.r.t. center
    Tensor<xpu, 2, DType> workspace = ctx.requested[center_loss_enum::kTempSpace].get_space_typed<xpu, 2, DType>(
        Shape2(param_.num_classes, k), s);
    DType alpha = param_.alpha;
    CenterLossBackward(diff, label, center, workspace, alpha);
    // gradient w.r.t. data
    DType grad_scale = param_.scale / n;
    Assign(grad, req[center_loss_enum::kData], diff*ScalarExp<DType>(grad_scale));
  }

 private:
  CenterLossParam param_;
};  // class CenterLossOp

template<typename xpu>
Operator *CreateOp(CenterLossParam param, int dtype);

#if DMLC_USE_CXX11
class CenterLossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    // name it bias, let Python code initialize it with 0
    return {"center_bias"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(center_loss_enum::kData);
    const TShape &lshape = in_shape->at(center_loss_enum::kLabel);
    if (dshape.ndim() == 0) return false;
    if (lshape.ndim() != 1) return false;
    CHECK_EQ(dshape.ndim(), 2) << "data: [BatchSize, FeatureDim]";
    index_t feature_dim = dshape[1];
    out_shape->clear();
    out_shape->push_back(dshape);
    aux_shape->clear();
    aux_shape->push_back(Shape2(param_.num_classes, feature_dim));
    return true;
  }

  std::string TypeString() const override {
    return "CenterLoss";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new CenterLossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const override {
    return {{in_data[center_loss_enum::kData], out_data[center_loss_enum::kOut]}};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const override {
    return {{out_data[center_loss_enum::kOut], in_grad[center_loss_enum::kData]}};
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {in_data[center_loss_enum::kLabel], out_data[center_loss_enum::kOut]};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  };

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  CenterLossParam param_;
};  // class CenterLossProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPREATEOR_CENTER_LOSS_INL_H_
