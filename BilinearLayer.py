import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def nmodeproduct(x, w, mode):
    if mode == 2:
        x = torch.mm(x, w)
    else:
        x = torch.mm(w,x)
    return x

class BL(nn.Module):
    def __init__(self, output_dim, input_dim):
        """
        :param output_dim: output dimensions of 2D tensor, should be a list of len 2, e.g. [30,20]
        :param input_dim: input dimensions of 3D tensor, should be a list of len 3, e.g. [1000,40,10]
        """

        super(BL, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.W1 = nn.Parameter(torch.Tensor(output_dim[0], input_dim[0]), requires_grad=True)
        self.W2 = nn.Parameter(torch.Tensor(input_dim[1], output_dim[1]), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]), requires_grad=True)

    def forward(self, x):
        print(x.shape)
        W1X = torch.Tensor(x.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(0, x.shape[0]):
            W1X[i] = nmodeproduct(x[i], self.W1, 1)
        W1XW2 = torch.Tensor(x.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(0, x.shape[0]):
            W1XW2[i] = nmodeproduct(W1X[i], self.W2, 2)
        for i in range(0, x.shape[0]):
            W1XW2[i] = W1XW2[i] + self.bias

        out = torch.Tensor(x.shape[0], self.output_dim[0])
        if self.output_dim[1] == 1:
            for i in range(0, x.shape[0]):
                out[i]=torch.squeeze(W1XW2[i],-1)
            return out

        return W1XW2

# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40, 10], [120, 5], [3, 1]]

# random data
x = np.random.rand(1000, 40, 10)
x = torch.from_numpy(x)
input = x.type(torch.LongTensor)


label = np.random.randint(0, 3, (1000, ))
item = torch.from_numpy(label)
ones = torch.sparse.torch.eye(3)
y = ones.index_select(0, item.long())
# target=y.type(torch.LongTensor)
target = y

loss_list = []
epochs = 10

model = BL(template[-1], [x.shape[1], x.shape[2]])

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.02)
# optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9)

for epoch in range(epochs):
    print("epoch:", epoch)
    output = model(input.float())
    output = torch.softmax(output,0)

    loss = criterion(output,target)
    print(loss.item())
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
























