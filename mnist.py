#!/usr/bin/env python2.7

import logging
import argparse
import mxnet as mx
import numpy as np

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from lsoftmax import LSoftmaxOp


logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_symbol():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=32)
    relu1 = mx.sym.Activation(data=conv1, act_type='relu')
    pool1 = mx.sym.Pooling(data=relu1, kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=64)
    relu2 = mx.sym.Activation(data=conv2, act_type='relu')
    pool2 = mx.sym.Pooling(data=relu2, kernel=(2, 2), stride=(2, 2), pool_type='max')
    fc3 = mx.sym.FullyConnected(data=pool2, num_hidden=256)
    relu3 = mx.sym.Activation(data=fc3, act_type='relu')

    embedding = mx.sym.FullyConnected(data=relu3, num_hidden=2, name='embedding')
    if args.lsoftmax:
        fc4 = mx.sym.Custom(data=embedding, label=label, num_hidden=10,
                            beta=args.beta, margin=args.margin, op_type='LSoftmax')
    else:
        fc4 = mx.sym.FullyConnected(data=embedding, num_hidden=10, no_bias=True)
    softmax_loss = mx.sym.SoftmaxOutput(data=fc4, label=label)
    return softmax_loss


def train():
    dev = mx.gpu(args.gpu) if args.gpu >=0 else mx.cpu()
    train = mx.io.MNISTIter(
                image='data/train-images-idx3-ubyte',
                label='data/train-labels-idx1-ubyte',
                input_shape=(1, 28, 28),
                mean_r=128,
                scale=1./128,
                batch_size=args.batch_size,
                shuffle=True)
    val = mx.io.MNISTIter(
                image='data/t10k-images-idx3-ubyte',
                label='data/t10k-labels-idx1-ubyte',
                input_shape=(1, 28, 28),
                mean_r=128,
                scale=1./128,
                batch_size=args.batch_size)
    symbol = get_symbol()
    model = mx.mod.Module(
                symbol=symbol,
                context=dev,
                data_names=('data',),
                label_names=('softmax_label',))
    num_examples = 60000
    epoch_size = int(num_examples / args.batch_size)
    optim_params = {
        'learning_rate': args.lr,
        'momentum': 0.9,
        'wd': 0.0005,
        'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=10*epoch_size, factor=0.1),
    }
    model.fit(train_data=train,
              eval_data=val,
              eval_metric=mx.metric.Accuracy(),
              initializer=mx.init.Xavier(),
              optimizer='sgd',
              optimizer_params=optim_params,
              num_epoch=args.num_epoch,
              batch_end_callback=mx.callback.Speedometer(args.batch_size, 50),
              epoch_end_callback=mx.callback.do_checkpoint(args.model_prefix))


def test():
    dev = mx.gpu(args.gpu) if args.gpu >=0 else mx.cpu()
    val = mx.io.MNISTIter(
            image='data/t10k-images-idx3-ubyte',
            label='data/t10k-labels-idx1-ubyte',
            input_shape=(1, 28, 28),
            batch_size=1)
    model = mx.model.FeedForward.load(args.model_prefix, args.num_epoch, ctx=dev)
    embedding = model.symbol.get_internals()['embedding_output']
    model = mx.model.FeedForward(ctx=dev, symbol=embedding, arg_params=model.arg_params,
                aux_params=model.aux_params, allow_extra_params=True, batch_size=1)
    embeds = []
    labels = []
    for batch in val:
        preds = model.predict(batch.data[0])
        embeds.append(preds[0])
        labels.append(batch.label[0].asnumpy())
    embeds = np.vstack(embeds)
    labels = np.hstack(labels)
    # vis, plot code from https://github.com/pangyupo/mxnet_center_loss
    num = len(labels)
    names = dict()
    for i in range(10):
        names[i]=str(i)
    palette = np.array(sns.color_palette("hls", 10))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(embeds[:,0], embeds[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(embeds[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, names[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig('mnist.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help="gpu device")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--beta', type=float, default=1, help="beta as lambda in lsoftmax")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--train', action='store_true', help="train mnist")
    parser.add_argument('--test', action='store_true', help="test mnist and plot")
    parser.add_argument('--lsoftmax', action='store_true', help="use lsoftmax layer")
    parser.add_argument('--margin', type=int, default=2, help="lsoftmax margin")
    parser.add_argument('--model-prefix', type=str, default='model/mnist', help="model predix")
    parser.add_argument('--num-epoch', type=int, default=20, help="number of epoches to train")
    args = parser.parse_args()
    print args

    if args.train:
        train()
    if args.test:
        test()
