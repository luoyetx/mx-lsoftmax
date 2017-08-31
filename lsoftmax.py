import os
import math
import mxnet as mx
import numpy as np


# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'


class LSoftmaxOp(mx.operator.CustomOp):
    '''LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>
    '''

    def __init__(self, margin, beta, beta_min, scale):
        self.margin = int(margin)
        self.beta = float(beta)
        self.beta_min = float(beta_min)
        self.scale = float(scale)
        self.c_map = []
        self.k_map = []
        c_m_n = lambda m, n: math.factorial(n) / math.factorial(m) / math.factorial(n-m)
        for i in range(margin+1):
            self.c_map.append(c_m_n(i, margin))
            self.k_map.append(math.cos(i * math.pi / margin))

    def find_k(self, cos_t):
        '''find k for cos(theta)
        '''
        # for numeric issue
        eps = 1e-5
        le = lambda x, y: x < y or abs(x-y) < eps
        for i in range(self.margin):
            if le(self.k_map[i+1], cos_t) and le(cos_t, self.k_map[i]):
                return i
        raise ValueError('can not find k for cos_t = %f'%cos_t)

    def calc_cos_mt(self, cos_t):
        '''calculate cos(m*theta)
        '''
        cos_mt = 0
        sin2_t = 1 - cos_t * cos_t
        flag = -1
        for p in range(self.margin / 2 + 1):
            flag *= -1
            cos_mt += flag * self.c_map[2*p] * pow(cos_t, self.margin-2*p) * pow(sin2_t, p)
        return cos_mt

    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 3
        assert len(out_data) == 1
        assert len(req) == 1
        x, label, w = in_data
        x = x.asnumpy()
        w =  w.asnumpy()
        label = label.asnumpy()
        # original fully connected
        out = x.dot(w.T)
        if is_train:
            # large margin fully connected
            n = label.shape[0]
            w_norm = np.linalg.norm(w, axis=1)
            x_norm = np.linalg.norm(x, axis=1)
            for i in range(n):
                j = yi = int(label[i])
                f = out[i, yi]
                cos_t = f / (w_norm[yi] * x_norm[i])
                # calc k and cos_mt
                k = self.find_k(cos_t)
                cos_mt = self.calc_cos_mt(cos_t)
                # f_i_j = (\beta * f_i_j + fo_i_j) / (1 + \beta)
                fo_i_j = f
                f_i_j = (pow(-1, k) * cos_mt - 2*k) * (w_norm[yi] * x_norm[i])
                out[i, yi] = (f_i_j + self.beta * fo_i_j) / (1 + self.beta)
        self.assign(out_data[0], req[0], mx.nd.array(out))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert len(in_data) == 3
        assert len(out_grad) == 1
        assert len(in_grad) == 3
        assert len(req) == 3
        x, label, w = in_data
        x = x.asnumpy()
        w = w.asnumpy()
        label = label.asnumpy()
        o_grad = out_grad[0].asnumpy()
        # original fully connected
        x_grad = o_grad.dot(w)
        w_grad = o_grad.T.dot(x)
        # large margin fully connected
        n = label.shape[0]  # batch size
        m = w.shape[0]  # number of classes
        margin = self.margin  # margin
        feature_dim = w.shape[1]  # feature dimension
        cos_t = np.zeros(n, dtype=np.float32)  # cos(theta)
        cos_mt = np.zeros(n, dtype=np.float32)  # cos(margin * theta)
        sin2_t = np.zeros(n, dtype=np.float32)  # sin(theta) ^ 2
        fo = np.zeros(n, dtype=np.float32)  # fo_i = dot(x_i, w_yi)
        k = np.zeros(n, dtype=np.int32)
        x_norm = np.linalg.norm(x, axis=1)
        w_norm = np.linalg.norm(w, axis=1)
        for i in range(n):
            j = yi = int(label[i])
            f = w[yi].dot(x[i])
            cos_t[i] = f / (w_norm[yi] * x_norm[i])
            k[i] = self.find_k(cos_t[i])
            cos_mt[i] = self.calc_cos_mt(cos_t[i])
            sin2_t[i] = 1 - cos_t[i]*cos_t[i]
            fo[i] = f
        # gradient w.r.t. x_i
        for i in range(n):
            # df / dx at x = x_i, w = w_yi
            j = yi = int(label[i])
            dcos_dx = w[yi] / (w_norm[yi]*x_norm[i]) - x[i] * fo[i] / (w_norm[yi]*pow(x_norm[i], 3))
            dsin2_dx = -2 * cos_t[i] * dcos_dx
            dcosm_dx = margin*pow(cos_t[i], margin-1) * dcos_dx  # p = 0
            flag = 1
            for p in range(1, margin / 2 + 1):
                flag *= -1
                dcosm_dx += flag * self.c_map[2*p] * ( \
                                p*pow(cos_t[i], margin-2*p)*pow(sin2_t[i], p-1)*dsin2_dx + \
                                (margin-2*p)*pow(cos_t[i], margin-2*p-1)*pow(sin2_t[i], p)*dcos_dx)
            df_dx = (pow(-1, k[i]) * cos_mt[i] - 2*k[i]) * w_norm[yi] / x_norm[i] * x[i] + \
                     pow(-1, k[i]) * w_norm[yi] * x_norm[i] * dcosm_dx
            alpha = 1 / (1 + self.beta)
            x_grad[i] += alpha * o_grad[i, yi] * (df_dx - w[yi])
        # gradient w.r.t. w_j
        for j in range(m):
            dw = np.zeros(feature_dim, dtype=np.float32)
            for i in range(n):
                yi = int(label[i])
                if yi == j:
                    # df / dw at x = x_i, w = w_yi and yi == j
                    dcos_dw = x[i] / (w_norm[yi]*x_norm[i]) - w[yi] * fo[i] / (x_norm[i]*pow(w_norm[yi], 3))
                    dsin2_dw = -2 * cos_t[i] * dcos_dw
                    dcosm_dw = margin*pow(cos_t[i], margin-1) * dcos_dw  # p = 0
                    flag = 1
                    for p in range(1, margin / 2 + 1):
                        flag *= -1
                        dcosm_dw += flag * self.c_map[2*p] * ( \
                                        p*pow(cos_t[i], margin-2*p)*pow(sin2_t[i], p-1)*dsin2_dw + \
                                        (margin-2*p)*pow(cos_t[i], margin-2*p-1)*pow(sin2_t[i], p)*dcos_dw)
                    df_dw_j = (pow(-1, k[i]) * cos_mt[i] - 2*k[i]) * x_norm[i] / w_norm[yi] * w[yi] + \
                               pow(-1, k[i]) * w_norm[yi] * x_norm[i] * dcosm_dw
                    dw += o_grad[i, yi] * (df_dw_j - x[i])
            alpha = 1 / (1 + self.beta)
            w_grad[j] += alpha * dw
        self.assign(in_grad[0], req[0], mx.nd.array(x_grad))
        self.assign(in_grad[2], req[2], mx.nd.array(w_grad))
        # dirty hack, should also work for multi devices
        self.beta *= self.scale
        self.beta = max(self.beta, self.beta_min)


@mx.operator.register("LSoftmax")
class LSoftmaxProp(mx.operator.CustomOpProp):

    def __init__(self, num_hidden, beta, margin, scale=1, beta_min=0):
        super(LSoftmaxProp, self).__init__(need_top_grad=True)
        self.margin = int(margin)
        self.num_hidden = int(num_hidden)
        self.beta = float(beta)
        self.beta_min = float(beta_min)
        self.scale = float(scale)

    def list_arguments(self):
        return ['data', 'label', 'weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert len(in_shape) == 3, "LSoftmaxOp input data: [data, label, weight]"
        dshape = in_shape[0]
        lshape = in_shape[1]
        assert len(dshape) == 2, "data shape should be (batch_size, feature_dim)"
        assert len(lshape) == 1, "label shape should be (batch_size,)"
        wshape = (self.num_hidden, dshape[1])
        oshape = (dshape[0], self.num_hidden)
        return [dshape, lshape, wshape], [oshape,], []

    def infer_type(self, in_type):
        return [in_type[0]]*len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return LSoftmaxOp(margin=self.margin, beta=self.beta, beta_min=self.beta_min, scale=self.scale)


def test_op():
    """test LSoftmax Operator
    """
    # build symbol
    batch_size = cmd_args.batch_size
    embedding_dim = cmd_args.embedding_dim
    num_classes = cmd_args.num_classes
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    weight = mx.sym.Variable('weight')
    args = {
        'data': np.random.normal(0, 1, (batch_size, embedding_dim)),
        'weight': np.random.normal(0, 1, (num_classes, embedding_dim)),
        'label': np.random.choice(num_classes, batch_size),
    }

    if cmd_args.op_impl == 'py':
        symbol = mx.sym.Custom(data=data, label=label, weight=weight, num_hidden=10,
                               beta=cmd_args.beta, margin=cmd_args.margin, scale=cmd_args.scale,
                               op_type='LSoftmax', name='lsoftmax')
    else:
        symbol = mx.sym.LSoftmax(data=data, label=label, weight=weight, num_hidden=num_classes,
                                 margin=cmd_args.margin, beta=cmd_args.beta, scale=cmd_args.scale,
                                 name='lsoftmax')

    data_shape = (batch_size, embedding_dim)
    label_shape = (batch_size,)
    weight_shape = (num_classes, embedding_dim)
    ctx = mx.cpu() if cmd_args.op_impl == 'py' else mx.gpu()
    executor = symbol.simple_bind(ctx=ctx, data=data_shape, label=label_shape, weight=weight_shape)

    def forward(data, label, weight):
        data = mx.nd.array(data, ctx=ctx)
        label = mx.nd.array(label, ctx=ctx)
        weight = mx.nd.array(weight, ctx=ctx)
        executor.forward(is_train=True, data=data, label=label, weight=weight)
        return executor.output_dict['lsoftmax_output'].asnumpy()

    def backward(out_grad):
        executor.backward(out_grads=[mx.nd.array(out_grad, ctx=ctx)])
        return executor.grad_dict

    def gradient_check(name, i, j):
        '''gradient check on x[i, j]
        '''
        eps = 1e-4
        threshold = 1e-2
        reldiff = lambda a, b: abs(a-b) / (abs(a) + abs(b))
        # calculate by backward
        output = forward(data=args['data'], weight=args['weight'], label=args['label'])
        grad_dict = backward(output)
        grad = grad_dict[name].asnumpy()[i, j]
        # calculate by \delta f / 2 * eps
        loss = lambda x: np.square(x).sum() / 2
        args[name][i, j] -= eps
        loss1 = loss(forward(data=args['data'], weight=args['weight'], label=args['label']))
        args[name][i, j] += 2 * eps
        loss2 = loss(forward(data=args['data'], weight=args['weight'], label=args['label']))
        grad_expect = (loss2 - loss1) / (2 * eps)
        # check
        rel_err = reldiff(grad_expect, grad)
        if rel_err > threshold:
            print 'gradient check failed'
            print 'expected %lf given %lf, relative error %lf'%(grad_expect, grad, rel_err)
            return False
        else:
            print 'gradient check pass'
            return True

    # test forward
    output = forward(data=args['data'], weight=args['weight'], label=args['label'])
    diff = args['data'].dot(args['weight'].T) - output

    # test backward
    # gradient check on data
    data_gc_pass = 0
    for i in range(args['data'].shape[0]):
        for j in range(args['data'].shape[1]):
            print 'gradient check on data[%d, %d]'%(i, j)
            if gradient_check('data', i, j):
                data_gc_pass += 1
    # gradient check on weight
    weight_gc_pass = 0
    for i in range(args['weight'].shape[0]):
        for j in range(args['weight'].shape[1]):
            print 'gradient check on weight[%d, %d]'%(i, j)
            if gradient_check('weight', i, j):
                weight_gc_pass += 1
    print '===== Summary ====='
    print 'gradient on data pass ratio is %lf'%(float(data_gc_pass) / args['data'].size)
    print 'gradient on weight pass ratio is %lf'%(float(weight_gc_pass) / args['weight'].size)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help="test batch size")
    parser.add_argument('--num-classes', type=int, default=10, help="test number of classes")
    parser.add_argument('--embedding-dim', type=int, default=3, help="test embedding dimension")
    parser.add_argument('--margin', type=int, default=2, help="test lsoftmax margin")
    parser.add_argument('--beta', type=float, default=10, help="test lsoftmax beta")
    parser.add_argument('--scale', type=float, default=1, help="beta scale of every mini-batch")
    parser.add_argument('--op-impl', type=str, choices=['py', 'cpp'], default='py', help="test op implementation")
    cmd_args = parser.parse_args()
    print cmd_args

    # check
    if cmd_args.op_impl == 'cpp':
        try:
            op_creator = mx.sym.LSoftmax
        except AttributeError:
            print 'No cpp operator for LSoftmax, Skip test'
            import sys
            sys.exit(0)

    test_op()
