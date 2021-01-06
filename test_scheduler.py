'''
Author: Di Li
Date: 2020-11-06 22:48:45
LastEditTime: 2020-11-07 17:06:39
LastEditors: Di Li
Description: 
FilePath: \EfficientNets_ddl_apex\test_scheduler.py
'''
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lr_scheduler import StepLR


if __name__ == '__main__':
    N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs.
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )

    #v = torch.zeros(10)
    optim = torch.optim.RMSprop(model.parameters(), lr=0.01)
    max_epoch = 100
    #optim = torch.optim.SGD([v], lr=0.01)
    scheduler = StepLR(optimizer=optim, step_size=24, warmup_t=30)


    x = []
    y = []
    for epoch in range(1, max_epoch):
        scheduler.step(epoch)
        x.append(epoch)
        y.append(optim.param_groups[0]['lr'])
        print(optim.param_groups[0]['lr'])
        #print(epoch, optim.param_groups[0]['lr'])

    #fig = plt.figure()
    #fig.plot(x,y)
    plt.scatter(x, y, color='red')
    plt.show()