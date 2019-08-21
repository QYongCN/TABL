import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TABL(nn.Module):
    def __init__(self, output_dim, input_dim, template):
        super(TABL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.template = template
        self.W1_1 = nn.Parameter(torch.Tensor(template[1][0], input_dim[0]), requires_grad=True)
        self.W1_2 = nn.Parameter(torch.Tensor(output_dim[0], template[1][0]), requires_grad=True)
        self.W_1 = nn.Parameter(torch.Tensor(input_dim[1], input_dim[1]), requires_grad=True)
        self.W_2 = nn.Parameter(torch.Tensor(template[1][1], template[1][1]), requires_grad=True)
        self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad = True)
        self.W2_1 = nn.Parameter(torch.Tensor(input_dim[1], template[1][1]), requires_grad=True)
        self.W2_2 = nn.Parameter(torch.Tensor(template[1][1], output_dim[1]), requires_grad=True)
        self.bias_1 = nn.Parameter(torch.Tensor(template[1][0], template[1][1]), requires_grad=True)
        self.bias_2 = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]), requires_grad=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        print(x.shape)

        #x-
        item1 = torch.Tensor(x.shape[0], template[1][0], self.input_dim[1])
        for i in range(x.shape[0]):
            item1[i] = self.W1_1@x[i]

        #E
        item2 = torch.Tensor(x.shape[0], template[1][0], self.input_dim[1])
        for i in range(x.shape[0]):
            item2[i] = item1[i]@self.W_1

        #A
        item2 = self.softmax(item2)

        item3 = torch.Tensor(x.shape[0], template[1][0], self.input_dim[1])
        for i in range(x.shape[0]):
            item3[i] = item1[i]*item2[i]

        #x~
        item4 = torch.Tensor(x.shape[0], template[1][0], self.input_dim[1])
        for i in range(x.shape[0]):
            item4[i] = item3[i]*self.alpha + item1[i]*(1-self.alpha)

        #y
        item5 = torch.Tensor(x.shape[0], template[1][0], template[1][1])
        for i in range(x.shape[0]):
            item5[i] = item4[i]@self.W2_1

        for i in range(x.shape[0]):
            item5[i] = item5[i] + self.bias_1

        item5 = self.relu(item5)

        # x-
        item6 = torch.Tensor(x.shape[0], self.output_dim[0], template[1][1])
        for i in range(x.shape[0]):
            item6[i] = self.W1_2 @ item5[i]

        # E
        item7 = torch.Tensor(x.shape[0], self.output_dim[0], template[1][1])
        for i in range(x.shape[0]):
            item7[i] = item6[i] @ self.W_2

        # A
        item7 = self.softmax(item7)

        item8 = torch.Tensor(x.shape[0], self.output_dim[0], template[1][1])
        for i in range(x.shape[0]):
            item8[i] = item6[i] * item7[i]

        # x~
        item9 = torch.Tensor(x.shape[0], self.output_dim[0], template[1][1])
        for i in range(x.shape[0]):
            item9[i] = item8[i] * self.alpha + item6[i] * (1 - self.alpha)

        # y
        item10 = torch.Tensor(x.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(x.shape[0]):
            item10[i] = item9[i] @ self.W2_2

        for i in range(x.shape[0]):
            item10[i] = item10[i] + self.bias_2

        out = torch.Tensor(x.shape[0], self.output_dim[0])

        if self.output_dim[1] == 1:
            for i in range(0, x.shape[0]):
                out[i]=torch.squeeze(item10[i],-1)
            return out

        return item10

# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40, 10], [120, 5], [3, 1]]

# random data
x = np.random.rand(1000, 40, 10)
x = torch.from_numpy(x)
input = x.type(torch.DoubleTensor)
input_mean, input_std = torch.mean(x, dim=0), torch.std(x, dim=0)
input_norm = (input - input_mean) / input_std

label = np.random.randint(0, 3, (1000, ))
item=torch.from_numpy(label)
ones = torch.sparse.torch.eye(3)
y = ones.index_select(0, item.long())
# target=y.type(torch.LongTensor)
target=y

loss_list = []
epochs = 20

model = TABL(template[-1], [x.shape[1], x.shape[2]], template)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

# criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.03)

for epoch in range(epochs):
    print("epoch:", epoch)
    output = model(input_norm.float()*10)

    loss = criterion(output,target)
    print(loss.item())
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



