import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class BL(nn.Module):
    def __init__(self, output_dim, input_dim, template):
        """
        :param output_dim: output dimensions of 2D tensor, should be a list of len 2, e.g. [30,20]
        :param input_dim: input dimensions of 3D tensor, should be a list of len 3, e.g. [1000,40,10]
        """

        super(BL, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.template = template
        self.W11 = nn.Parameter(torch.Tensor(template[1][0], input_dim[0]), requires_grad=True)
        self.W12 = nn.Parameter(torch.Tensor(output_dim[0], template[1][0]), requires_grad=True)
        self.fc1 = nn.Linear(self.input_dim[0], self.template[1][0], bias=False)
        self.fc2 = nn.Linear(self.input_dim[1], self.template[1][1],  bias=False)
        self.fc3 = nn.Linear(self.template[1][0], self.output_dim[0], bias=False)
        self.fc4 = nn.Linear(self.template[1][1], self.output_dim[1], bias=False)
        self.bias1 = nn.Parameter(torch.Tensor(template[1][0], template[1][1]), requires_grad=True)
        self.bias2 = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]), requires_grad=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
       print(x.shape)

       item1=torch.Tensor(x.shape[0], self.template[1][0], self.input_dim[1])
       for i in range(x.shape[0]):
           item1[i] = self.W11@x[i]

       item2 = torch.Tensor(x.shape[0], self.template[1][0], self.template[1][1])
       for i in range(x.shape[0]):
           item2[i] = self.fc2(item1[i])

       for i in range(x.shape[0]):
           item2[i] = item2[i] + self.bias1

       item2 = self.relu(item2)
       item2 = self.dropout(item2)

       item3 = torch.Tensor(x.shape[0], self.output_dim[0], self.template[1][1])
       for i in range(x.shape[0]):
           item3[i] = self.W12@item2[i]

       item4 = torch.Tensor(x.shape[0], self.output_dim[0], self.output_dim[1])
       for i in range(x.shape[0]):
           item4[i] = self.fc4(item3[i])

       for i in range(x.shape[0]):
           item4[i] = item4[i] + self.bias2

       out = torch.Tensor(x.shape[0], self.output_dim[0])
       if self.output_dim[1] == 1:
           for i in range(0, x.shape[0]):
               out[i] = torch.squeeze(item4[i], -1)
           return out

       return item4




# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40, 10], [120, 5], [3, 1]]

# random data
x = np.random.rand(1000, 40, 10)
x = torch.from_numpy(x)
input = x.type(torch.DoubleTensor)
input_mean, input_std = torch.mean(x, dim=0), torch.std(x, dim=0)
input_norm = (input - input_mean) / input_std


label = np.random.randint(0, 3, (1000, ))
item = torch.from_numpy(label)
ones = torch.sparse.torch.eye(3)
y = ones.index_select(0, item.long())
# target=y.type(torch.LongTensor)
target = y

loss_list = []
epochs = 3

model = BL(template[-1],template[0],template)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

# criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr = 0.02)
optimizer = optim.SGD(model.parameters(), lr=0.0000002, momentum=0.9)

for epoch in range(epochs):
    print("epoch:",epoch)
    # output=model(x.float())
    output=model(input_norm.float())

    loss = criterion(output, target)
    print(loss.item())
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
























