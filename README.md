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

I implement the operator both in Python and C++(CUDA). The performance below is training LeNet on a single GTX1070 with parameters margin = 3, lambda = 1. **Notice** the C++ implement can only run on GPU context.

|Batch Size     |traditional fully connected    |lsoftmax in Python         |lsoftmax in C++(CUDA)      |
|---------------|-------------------------------|---------------------------|---------------------------|
|128            |~45000 samples / sec           |3000 ~ 4000 samples / sec  |~26000 samples / sec       |
|256            |~54000 samples / sec           |5000 ~ 6000 samples / sec  |~30000 samples / sec       |

## Visualization

### original softmax (traditional fully connected)

![lsoftmax-margin-1](imgs/mnist-lsoftmax-margin-1.png)

### lsoftmax with margin = 2 and lambda = 1

![lsoftmax-margin-2](imgs/mnist-lsoftmax-margin-2.png)

### lsoftmax with margin = 3 and lambda = 1

![lsoftmax-margin-3](imgs/mnist-lsoftmax-margin-3.png)

### lsoftmax with margin = 4 and lambda = 1

![lsoftmax-margin-4](imgs/mnist-lsoftmax-margin-4.png)

## References

- [MXNet](mxnet)
- [mxnet_center_loss](mxnet-center-loss)
- [Large-Margin Softmax Loss for Convolutional Neural Networks][lsoftmax]


[mxnet]: https://github.com/dmlc/mxnet
[lsoftmax]: https://arxiv.org/pdf/1612.02295.pdf
[mxnet-center-loss]: https://github.com/pangyupo/mxnet_center_loss
