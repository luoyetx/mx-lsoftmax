#!/usr/bin/env bash

echo "Copy files to $MXNET_HOME"

cp ./operator/lsoftmax-inl.h $MXNET_HOME/src/operator/lsoftmax-inl.h
cp ./operator/lsoftmax.cc $MXNET_HOME/src/operator/lsoftmax.cc
cp ./operator/lsoftmax.cu $MXNET_HOME/src/operator/lsoftmax.cu
cp ./operator/center_loss-inl.h $MXNET_HOME/src/operator/center_loss-inl.h
cp ./operator/center_loss.cc $MXNET_HOME/src/operator/center_loss.cc
cp ./operator/center_loss.cu $MXNET_HOME/src/operator/center_loss.cu
