mx-lsoftmax
===========

mxnet version of [Large-Margin Softmax Loss for Convolutional Neural Networks][lsoftmax].

## Derivatives

I put all formula I used to calculate the derivatives below. You can check it by yourself. If there's a mistake, please do tell me or open an issue.

The derivatives doesn't include `lambda` in the paper, but the code does.

![formula](imgs/formula.jpg)

## Gradient Check

Gradient check can be failed with data type float32 but ok with data type float64. So don't afraid to see gradient check failed.

## Operator Performance

Currently I only implement the operator in Python. I will implement it in C++ soon. The performance in Python is really poor. The speed is only 3000~4000 samples / sec vs using traditional fully connected op is 40000~50000 samples / sec when training LeNet on GPU.

## Visualization

### original softmax (traditional fully connected)

![lsoftmax-margin-1](imgs/mnist-lsoftmax-margin-1.jpg)

### lsoftmax with margin = 2 and lambda = 1

![lsoftmax-margin-2](imgs/mnist-lsoftmax-margin-2.jpg)

### lsoftmax with margin = 3 and lambda = 1

![lsoftmax-margin-3](imgs/mnist-lsoftmax-margin-3.jpg)

### lsoftmax with margin = 4 and lambda = 1

![lsoftmax-margin-4](imgs/mnist-lsoftmax-margin-4.jpg)

## References

- [MXNet](mxnet)
- [mxnet_center_loss](mxnet-center-loss)
- [Large-Margin Softmax Loss for Convolutional Neural Networks][lsoftmax]


[mxnet]: https://github.com/dmlc/mxnet
[lsoftmax]: https://arxiv.org/pdf/1612.02295.pdf
[mxnet-center-loss]: https://github.com/pangyupo/mxnet_center_loss
