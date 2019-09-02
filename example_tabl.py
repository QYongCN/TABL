import Layer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

epochs = 1000

model_1 = Layer.TABL(template[1], template[0])
model_2 = Layer.TABL(template[-1], template[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model_2.parameters(), lr=0.01)

for epoch in range(epochs):
    print("epoch:",epoch)
    # output=model(x.float())
    output=model_1(input_norm.float())
    out = model_2(output)

    loss = criterion(out, target)
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()