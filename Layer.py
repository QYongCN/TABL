import torch
import torch.nn as nn

class BL(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(BL, self).__init__()
        self.output_dim = output_dim
        self.input_dium = input_dim

        self.W1 = nn.Parameter(torch.Tensor(output_dim[0], input_dim[0]))
        self.W2 = nn.Parameter(torch.Tensor(input_dim[1], output_dim[1]))
        self.bias = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]))

    def forward(self, X):
        print(X.shape)

        left_multiplication = torch.Tensor(X.shape[0], self.output_dim[0], self.input_dium[1])
        for i in range(X.shape[0]):
            left_multiplication[i] = self.W1 @ X[i]

        right_multiplication = torch.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(X.shape[0]):
            right_multiplication[i] = left_multiplication[i] @ self.W2

        Y = torch.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(X.shape[0]):
            Y[i] = right_multiplication[i] + self.bias

        out = torch.Tensor(X.shape[0], self.output_dim[0])
        if self.output_dim[1] == 1:
            for i in range(0, X.shape[0]):
                out[i] = torch.squeeze(Y[i], -1)
            return out

        return Y

class TABL(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(TABL, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W1 = nn.Parameter(torch.Tensor(output_dim[0], input_dim[0]))
        self.W = nn.Parameter(torch.Tensor(input_dim[1], input_dim[1]))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.W2 = nn.Parameter(torch.Tensor(input_dim[1], output_dim[1]))
        self.bias = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]))
        self.softmax = nn.Softmax()

    def forward(self, X):
        print(X.shape)

        X1 = torch.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(X.shape[0]):
            X1[i] = self.W1 @ X[i]

        E = torch.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(X.shape[0]):
            E[i] = X1[i] @ self.W

        A = torch.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1])
        A = self.softmax(E)

        Attention = torch.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(X.shape[0]):
            Attention[i] = X1[i] * A[i]

        tem = torch.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(X.shape[0]):
            tem[i] = self.alpha * Attention[i] + (1 - self.alpha) * X1[i]

        X2 = torch.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(X.shape[0]):
            X2[i] = tem[i] @ self.W2

        Y = torch.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(X.shape[0]):
            Y[i] = X2[i] + self.bias

        out = torch.Tensor(X.shape[0], self.output_dim[0])
        if self.output_dim[1] == 1:
            for i in range(0, X.shape[0]):
                out[i] = torch.squeeze(Y[i], -1)
            return out

        return Y


















