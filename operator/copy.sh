#!/usr/bin/env bash

echo "Copy files to $MXNET_HOME"

cp ./operator/lsoftmax-inl.h $MXNET_HOME/src/operator/lsoftmax-inl.h
cp ./operator/lsoftmax.cc $MXNET_HOME/src/operator/lsoftmax.cc
cp ./operator/lsoftmax.cu $MXNET_HOME/src/operator/lsoftmax.cu
