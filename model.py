import torch
from torch.nn import init
from layer import *

class DCGAN_G(nn.Module):
    def __init__(self, nch_in,n_class, nch_out, nch_ker=128):
        super(DCGAN_G, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.n_class = n_class

        self.DECBR1_1 = DECBR2D(1 * self.nch_in, 4 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        self.DECBR1_2 = DECBR2D(1 * self.n_class, 4 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=True, relu=0.0, bias=False)
        self.DECBR2 = DECBR2D(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR3 = DECBR2D(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR4 = DECBR2D(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.0, bias=False)
        self.DECBR5 = DECBR2D(1 * self.nch_ker, 1 * self.nch_out, kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)

    def forward(self, input, label):
        x_1 = self.DECBR1_1(input)
        x_2 = self.DECBR1_2(label)
        x = torch.cat([x_1,x_2],1)
        x = self.DECBR2(x)
        x = self.DECBR3(x)
        x = self.DECBR4(x)
        x = self.DECBR5(x)

        x = torch.tanh(x)
        return x

class DCGAN_D(nn.Module):
    def __init__(self, nch_in,n_class, nch_ker=64):
        super(DCGAN_D, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.n_class = n_class

        self.CBR1_1 = CBR2D(1 * nch_in, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR1_2 = CBR2D(1 * n_class, 1 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR2 = CBR2D(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR3 = CBR2D(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR4 = CBR2D(8 * nch_ker, 16 * nch_ker, kernel_size=4, stride=2, padding=1, norm=True, relu=0.2, bias=False)
        self.CBR5 = CBR2D(16 * nch_ker, 1,           kernel_size=4, stride=2, padding=1, norm=False, relu=[], bias=False)

    def forward(self,input, label):

        x_1 = self.CBR1_1(input)
        x_2 = self.CBR1_2(label)
        x = torch.cat([x_1,x_2],1)
        x = self.CBR2(x)
        x = self.CBR3(x)
        x = self.CBR4(x)
        x = self.CBR5(x)

        x = torch.sigmoid(x)

        return x

def init_weights(net, init_gain=0.02):

    def  init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and(classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
