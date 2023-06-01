"""
need improved to AdaptiveAvgPool or auto-adjested linear shape
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchinfo import summary

class customSNet(nn.Module):
    def __init__(
        self,
        num_steps,
        beta,
        input_fdim,
        input_tdim,
        threshold=1.0,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        num_class=10,
    ):
        super().__init__()
        self.fdim = input_fdim
        self.tdim = input_tdim
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc1 = nn.Linear(47040, 128)  # (47040, 128) for mfcc
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc2 = nn.Linear(128, 64)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc3 = nn.Linear(64, num_class)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

    def calc_flatten_fc_neuron(self):
        size = self.conv_block(torch.randn(1, 1, self.fdim, self.tdim)).size()
        m = 1
        for i in size:
            m *= i
        return int(m)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        batch_size_curr = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []

        for step in range(self.num_steps):
            # cur1 = self.pool(self.conv1(x))
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # cur2 = self.pool(self.conv2(spk1))
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.view(batch_size_curr, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc3(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        return torch.stack(spk5_rec), torch.stack(mem5_rec)
    
if __name__ == '__main__':
    batch_size = 32
    input_fdim = 32
    input_tdim = 109
    num_classes = 7

    snn_mdl = customSNet(num_steps=10, beta=0.5, input_fdim=input_fdim, input_tdim=input_tdim, num_class=num_classes)
    test_input = torch.rand([batch_size, 1, input_fdim, input_tdim])
    output = snn_mdl(test_input)
    print(output[0].shape)

    ## print(summary(snn_mdl.cuda(), input_size=(batch_size, 1, input_fdim, input_tdim)))
