import os
import math
import mxnet as mx
import numpy as np


# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'


class LSoftmaxOp(mx.operator.CustomOp):
    '''LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>
    '''

    def __init__(self, margin, beta):
        self.margin = int(margin)
        self.beta = float(beta)
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
        sin_t_2 = 1 - cos_t * cos_t
        flag = -1
        for p in range(self.margin / 2 + 1):
            flag *= -1
            cos_mt += flag * self.c_map[2*p] * pow(cos_t, self.margin-2*p) * pow(sin_t_2, p)
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
        # large margin fully connected
        n = label.shape[0]
        w_norm = np.linalg.norm(w, axis=1)
        x_norm = np.linalg.norm(x, axis=1)
        for i in range(n):
            j = yi = int(label[i])
            f = w[yi].dot(x[i])
            cos_t = f / (w_norm[yi] * x_norm[i])
            # calc k and cos_mt
            k = self.find_k(cos_t)
            cos_mt = self.calc_cos_mt(cos_t)
            # f_i_j = (\beta * f_i_j + fo_i_j) / (1 + \beta)
            fo_i_j = f
            f_i_j = (pow(-1, k) * cos_mt - 2*k) * (w_norm[yi] * x_norm[i])
            out[i, yi] = (self.beta * f_i_j + fo_i_j) / (1 + self.beta)
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
        # gradient w.r.t. x_i
        for i in range(n):
            # df / dx at x = x_i, w = w_yi
            j = yi = int(label[i])
            dcos_dx = w[yi] / (w_norm[yi]*x_norm[i]) - x[i] * w[yi].dot(x[i]) / (w_norm[yi]*pow(x_norm[i], 3))
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
            alpha = self.beta / (1 + self.beta)
            x_grad[i] += alpha * o_grad[i, yi] * (df_dx - w[yi])
        # gradient w.r.t. w_j
        for j in range(m):
            dw = np.zeros(feature_dim, dtype=np.float32)
            for i in range(n):
                yi = int(label[i])
                if yi == j:
                    # df / dw at x = x_i, w = w_yi and yi == j
                    dcos_dw = x[i] / (w_norm[yi]*x_norm[i]) - w[yi] * w[yi].dot(x[i]) / (x_norm[i]*pow(w_norm[yi], 3))
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
            alpha = self.beta / (1 + self.beta)
            w_grad[j] += alpha * dw
        self.assign(in_grad[0], req[0], mx.nd.array(x_grad))
        self.assign(in_grad[2], req[2], mx.nd.array(w_grad))


@mx.operator.register("LSoftmax")
class LSoftmaxProp(mx.operator.CustomOpProp):

    def __init__(self, num_hidden, beta, margin):
        super(LSoftmaxProp, self).__init__(need_top_grad=True)
        self.margin = int(margin)
        self.num_hidden = int(num_hidden)
        self.beta = float(beta)

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

    def create_operator(self, ctx, shapes, dtypes):
        return LSoftmaxOp(margin=self.margin, beta=self.beta)


def test_python_op():
    """test LSoftmax Operator implemented in Python
    """
    # build op
    batch_size = cmd_args.batch_size
    embedding_dim = cmd_args.embedding_dim
    num_classes = cmd_args.num_classes
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    weight = mx.sym.Variable('weight')
    x = np.random.normal(0, 1, (batch_size, embedding_dim))
    w = np.random.normal(0, 1, (num_classes, embedding_dim))
    y = np.random.choice(num_classes, batch_size)
    out = np.zeros((batch_size, num_classes))
    ctx = mx.cpu()
    args = {
        'data': mx.nd.array(x, ctx=ctx),
        'label': mx.nd.array(y, ctx=ctx),
        'weight': mx.nd.array(w, ctx=ctx),
        'output': mx.nd.array(out, ctx=ctx),
    }
    args_grad = {
        'data': mx.nd.zeros(x.shape, ctx=ctx),
        'label': mx.nd.zeros(y.shape, ctx=ctx),
        'weight': mx.nd.zeros(w.shape, ctx=ctx),
        'output': mx.nd.zeros(out.shape, ctx=ctx),
    }
    op = LSoftmaxOp(margin=cmd_args.margin, beta=cmd_args.beta)

    def forward():
        op.forward(is_train=True, req=['write'],
                   in_data=[args['data'], args['label'], args['weight'],],
                   out_data=[args['output'],],
                   aux=[])

    def backward():
        op.backward(req=['write', 'null', 'write'],
                    out_grad=[args_grad['output'],],
                    in_data=[args['data'], args['label'], args['weight'],],
                    out_data=[args['output'],],
                    in_grad=[args_grad['data'], args_grad['label'], args_grad['weight'],],
                    aux=[])

    # test forward
    forward()
    output = args['output'].asnumpy()
    diff = x.dot(w.T) - output

    # test backward
    loss = lambda x: np.square(x).sum() / 2

    def gradient_check(name, i, j):
        '''gradient check on x[i, j]
        '''
        eps = 1e-4
        threshold = 1e-2
        reldiff = lambda a, b: abs(a-b) / (abs(a) + abs(b))
        x = args[name].asnumpy()
        # calculate by backward
        forward()
        args_grad['output'] = args['output']
        backward()
        grad = args_grad[name].asnumpy()[i, j]
        # calculate by \delta f / 2 * eps
        x1 = x.copy()
        x1[i, j] -= eps
        args[name] = mx.nd.array(x1, ctx=ctx)
        forward()
        loss1 = loss(args['output'].asnumpy())
        x2 = x.copy()
        x2[i, j] += eps
        args[name] = mx.nd.array(x2, ctx=ctx)
        forward()
        loss2 = loss(args['output'].asnumpy())
        grad_ = (loss2 - loss1) / (2 * eps)
        # check
        rel_err = reldiff(grad_, grad)
        if rel_err > threshold:
            print 'gradient check failed'
            print 'expected %lf given %lf, relative error %lf'%(grad_, grad, rel_err)
        else:
            print 'gradient check pass'

    # gradient check on x
    n = x.size / 2
    for i in range(n):
        x_i, x_j = np.random.choice(x.shape[0]), np.random.choice(x.shape[1])
        print 'gradient check on x[%d, %d]'%(x_i, x_j)
        gradient_check('data', x_i, x_j)
    # gradient check on w
    n = w.size / 2
    for i in range(n):
        w_i, w_j = np.random.choice(w.shape[0]), np.random.choice(w.shape[1])
        print 'gradient check on w[%d, %d]'%(w_i, w_j)
        gradient_check('weight', w_i, w_j)


def test_cpp_op():
    """test LSoftmax Operator implemented in C++
    """
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help="test batch size")
    parser.add_argument('--num-classes', type=int, default=10, help="test number of classes")
    parser.add_argument('--embedding-dim', type=int, default=3, help="test embedding dimension")
    parser.add_argument('--margin', type=int, default=2, help="test lsoftmax margin")
    parser.add_argument('--beta', type=float, default=10, help="test lsoftmax beta")
    cmd_args = parser.parse_args()
    print cmd_args

    test_python_op()
    test_cpp_op()
