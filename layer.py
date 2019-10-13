import torch as tf
import torch.nn as nn

class BL(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(BL,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.W1 = nn.Parameter(tf.Tensor(output_dim[0], input_dim[0]))
        self.W2 = nn.Parameter(tf.Tensor(input_dim[1], output_dim[1]))
        self.b = nn.Parameter(tf.Tensor(output_dim[0], output_dim[1]))

    def forward(self, X):
        left_multiplication = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dium[1])
        for i in range(X.shape[0]):
            left_multiplication[i] = self.W1 @ X[i]

        right_multiplication = tf.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(X.shape[0]):
            right_multiplication[i] = left_multiplication[i] @ self.W2

        Y = tf.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(X.shape[0]):
            Y[i] = right_multiplication[i] + self.bias

        out = tf.Tensor(X.shape[0], self.output_dim[0])
        if self.output_dim[1] == 1:
            for i in range(0, X.shape[0]):
                out[i] = tf.squeeze(Y[i], -1)
            return out

        return Y

class TABL(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(TABL, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W1 = nn.Parameter(tf.Tensor(output_dim[0], input_dim[0]))
        self.W = nn.Parameter(tf.Tensor(input_dim[1], input_dim[1]))
        self.alpha = nn.Parameter(tf.Tensor(1)).cuda()
        self.W2 = nn.Parameter(tf.Tensor(input_dim[1], output_dim[1]))
        self.bias = nn.Parameter(tf.Tensor(output_dim[0], output_dim[1]))
        self.softmax = nn.Softmax()

    def forward(self, X):
        X1 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            X1[i] = self.W1 @ X[i]

        E = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            E[i] = X1[i] @ self.W

        A = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        A = self.softmax(E)

        Attention = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            Attention[i] = X1[i] * A[i]

        tem = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            tem[i] = self.alpha * Attention[i] + (1 - self.alpha) * X1[i]

        X2 = tf.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1]).cuda()
        for i in range(X.shape[0]):
            X2[i] = tem[i] @ self.W2

        Y = tf.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1]).cuda()
        for i in range(X.shape[0]):
            Y[i] = X2[i] + self.bias

        out = tf.Tensor(X.shape[0], self.output_dim[0]).cuda()
        if self.output_dim[1] == 1:
            for i in range(0, X.shape[0]):
                out[i] = tf.squeeze(Y[i], -1)
            return out

        return Y

# class DFSMN(nn.Module):
#     def __init__(self, output_dim, input_dim):
#         super(DFSMN, self).__init__()
#         self.output_dim = output_dim
#         self.input_dim = input_dim
#         self.W1 = nn.Parameter(tf.Tensor())
#         self.V1 = nn.Parameter(tf.Tensor())
#         self.U1 = nn.Parameter(tf.Tensor())
#         self.V2 = nn.Parameter(tf.Tensor())
#         self.U2 = nn.Parameter(tf.Tensor())
#         self.V3 = nn.Parameter(tf.Tensor())
#         self.U3 = nn.Parameter(tf.Tensor())
#         self.V4 = nn.Parameter(tf.Tensor())
#         self.U4 = nn.Parameter(tf.Tensor())
#         self.W5 = nn.Parameter(tf.Tensor())
#         self.W6 = nn.Parameter(tf.Tensor())
#         self.V7 = nn.Parameter(tf.Tensor())
#         self.U7 = nn.Parameter(tf.Tensor())
#
#     def forward(self, X):
#         h1 = tf.Tensor(X.shape[0], , )
#         for i in range(X.shape[0]):
#             h1[i] = X @ self.W1
#
#         p1 = tf.Tensor(X.shape[0], , )
#         for i in range(X.shape[0]):
#             p1[i] = h1 @ self.V1
#
#         p1_ = p1
#
#         h2 = tf.Tensor(X.shape[0], , )
#         for i in range(X.shape[0]):
#             h2[i] = p1_ @ self.U1
#
#         p2 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             p2[i] = h2 @ self.V2
#
#         p2_ = p2 +
#
#         h3 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             h3[i] = p2_ @ self.U2
#
#         p3 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             p3[i] = h3 @ self.V3
#
#         p3_ = p3 +
#
#         h4 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             h4[i] = p3_ @ self.U3
#
#         p4 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             p4[i] = h4 @ self.V4
#
#         p4_ = p4 +
#
#         h5 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             h5[i] = p4_ @ self.U4
#
#         h6 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             h6[i] = h5[i] @ self.W5
#
#         h7 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             h7[i] = h6[i] @ self.W6
#
#         p8 = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             p8[i] = h6[i] @ self.V7
#
#         y = tf.Tensor(X.shape[0],, )
#         for i in range(X.shape[0]):
#             y[i] = p8[i] @ self.U7
#
#         return y





















