"""
   crown.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.in_to_hid1 = nn.Linear(2, hid)
        self.hid1_to_hid2 = nn.Linear(hid, hid)
        self.hid2_to_out = nn.Linear(hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hid1 = None
        self.hid2 = None

    def forward(self, input):
        self.hid1 = self.tanh(self.in_to_hid1(input))
        self.hid2 = self.tanh(self.hid1_to_hid2(self.hid1))
        output = self.sigmoid(self.hid2_to_out(self.hid2))
        return output


class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.in_to_hid1 = nn.Linear(2, hid)
        self.hid1_to_hid2 = nn.Linear(hid, hid)
        self.hid2_to_hid3 = nn.Linear(hid, hid)
        self.hid3_to_out = nn.Linear(hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hid1 = None
        self.hid2 = None
        self.hid3 = None

    def forward(self, input):
        self.hid1 = self.tanh(self.in_to_hid1(input))
        self.hid2 = self.tanh(self.hid1_to_hid2(self.hid1))
        self.hid3 = self.tanh(self.hid2_to_hid3(self.hid2))
        output = self.sigmoid(self.hid3_to_out(self.hid3))
        return output


class DenseNet(torch.nn.Module):
    def __init__(self, hid):
        super(DenseNet, self).__init__()
        self.in_to_hid1 = nn.Linear(2, hid)
        self.in_to_hid2 = nn.Linear(2, hid)
        self.in_to_out = nn.Linear(2, 1)
        self.hid1_to_hid2 = nn.Linear(hid, hid)
        self.hid1_to_out = nn.Linear(hid, 1)
        self.hid2_to_out = nn.Linear(hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hid1 = None
        self.hid2 = None

    def forward(self, input):
        in_to_hid1 = self.in_to_hid1(input)
        in_to_hid2 = self.in_to_hid2(input)
        in_to_out = self.in_to_out(input)
        self.hid1 = self.tanh(in_to_hid1)
        hid1_to_hid2 = self.hid1_to_hid2(self.hid1)
        hid2_input = in_to_hid2 + hid1_to_hid2
        self.hid2 = self.tanh(hid2_input)
        hid1_to_out = self.hid1_to_out(self.hid1)
        hid2_to_out = self.hid2_to_out(self.hid2)
        output_input = in_to_out + hid1_to_out + hid2_to_out
        output = self.sigmoid(output_input)
        return output

        # self.hid1 = None
        # self.hid2 = None
        # return 0 * input[:, 0]
