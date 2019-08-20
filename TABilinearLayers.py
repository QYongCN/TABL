import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def nmodeproduct(x, w, mode):
    if mode == 2:
        x = torch.mm(x, w)
    else:
        x = torch.mm(w, x)
    return x

class TABL(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(TABL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W1 = nn.Parameter(torch.Tensor(output_dim[0], input_dim[0]), requires_grad = True)
        self.W2 = nn.Parameter(torch.Tensor(input_dim[1], output_dim[1]), requires_grad = True)
        self.W = nn.Parameter(torch.Tensor(input_dim[1], input_dim[1]), requires_grad = True)
        self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad = True)
        self.bias = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]), requires_grad = True)
        self.relu = nn.ReLU()


    def forward(self, x):
        #X [input[0],input[1]],W1 [output[0],input[0]]
        #W1X shape [output[0],input[1]]
        #x-
        item = torch.Tensor(x.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(0, x.shape[0]):
            item[i] = nmodeproduct(x[i], self.W1, 1)
        #W [input[1],input[1]]
        #W1XW shape [output[0],input[1]]
        #E
        item1 = torch.Tensor(x.shape[0], self.output_dim[0], self.input_dim[1])
        #W = self.W - self.W * torch.eye(self.input_dim[2], dtype='float32') + torch.eye(self.input_dim[2], dtype='float32') / self.input_dim[2]
        for i in range(0, x.shape[0]):
            item1 = nmodeproduct(item[i], self.W, 2)
        #A softmax shape[output[0],input[1]] attention
        #attention
        item2 = torch.Tensor(x.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(0, x.shape[0]):
            item2[i] = torch.softmax(item1[1], 0)
        #W1X shape [output[0],input[1]]   A shape[output[0],input[1]]
        #x-,A
        item3 = torch.Tensor(x.shape[0], self.output_dim[0], self.input_dim[1])
        for i in range(0, x.shape[0]):
            item3[i] = self.alpha*item[i] + (1.0 - self.alpha) * item[i] * item2[i]
        #x~
        item4 = torch.Tensor(x.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(0, x.shape[0]):
            item4[i] = nmodeproduct(item3[i], self.W2, 2) + self.bias
        item5 = torch.Tensor(x.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(0,x.shape[0]):
            item5[i] = torch.softmax(item4[i], 0)

        out = torch.Tensor(x.shape[0], self.output_dim[0])

        if self.output_dim[1] == 1:
            for i in range(0, x.shape[0]):
                out[i]=torch.squeeze(item5[i],-1)
            return out

        return item5

# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40, 10], [120, 5], [3, 1]]

# random data
x = np.random.rand(1000, 40, 10)
x = torch.from_numpy(x)
input = x.type(torch.LongTensor)

label = np.random.randint(0, 3, (1000, ))
item=torch.from_numpy(label)
ones = torch.sparse.torch.eye(3)
y = ones.index_select(0, item.long())
# target=y.type(torch.LongTensor)
target=y

loss_list = []
epochs = 20

model = TABL(template[-1], [x.shape[1], x.shape[2]])

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.02)

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



