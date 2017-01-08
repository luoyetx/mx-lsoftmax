#!/usr/bin/env python2.7

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_beta():
    '''plot beta over training
    '''
    beta = args.beta
    scale = args.scale
    beta_min = args.beta_min
    num_epoch = args.num_epoch
    epoch_size = int(float(args.num_examples) / args.batch_size)

    x = np.arange(num_epoch*epoch_size)
    y = beta * np.power(scale, x)
    y = np.maximum(y, beta_min)
    epoch_x = np.arange(num_epoch) * epoch_size
    epoch_y = beta * np.power(scale, epoch_x)
    epoch_y = np.maximum(epoch_y, beta_min)

    # plot beta descent curve
    plt.semilogy(x, y)
    plt.semilogy(epoch_x, epoch_y, 'ro')
    plt.title('beta descent')
    plt.ylabel('beta')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-examples', type=int, default=60000, help="number of training data")
    parser.add_argument('--batch-size', type=int, default=256, help="batch size of mini-batch")
    parser.add_argument('--beta', type=float, default=100, help="initial beta")
    parser.add_argument('--scale', type=float, default=0.99, help="scale in beta descent")
    parser.add_argument('--beta-min', type=float, default=1e-2, help="minimun beta during training")
    parser.add_argument('--num-epoch', type=int, default=20, help="number of epoches to train")
    args = parser.parse_args()
    print args

    plot_beta()
