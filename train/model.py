import sys
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

import torch
import torch.nn as nn

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size, bias=True)
        self.l2 = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.l1(x)
        output = self.tanh(output)
        output = self.l2(output)
        output = torch.softmax(output, dim=1)
        output = (x * output).sum(dim=1)
        return output
        
class Confirming_model(nn.Module):
    hparams = {
        "n_mels": 40,
        "cnn_channels": 16,
        "cnn_kernel_size": (20, 5),
        "stride": (8, 2),
        "gru_hidden_size": 64,
        "attention_hidden_size": 64,
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 50, 
        "test_size": 0.25,
        "n_classes": 2
    }

    def __init__(self, 
                 n_mels = 40,
                 cnn_channels = 16, 
                 cnn_kernel_size = (20, 5),
                 stride = (8, 2),
                 gru_hidden_size = 64, 
                 attention_hidden_size = 64,
                 n_classes = 0):
      
        super().__init__()
        self.cnn = nn.Conv2d(1, cnn_channels, kernel_size=cnn_kernel_size, stride=stride,
                             padding=(cnn_kernel_size[0]//2, cnn_kernel_size[1]//2))
        self.fully_connected = nn.Linear((n_mels//stride[0] + 1)*cnn_channels, cnn_channels)
        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden_size, 
                          bidirectional=True, batch_first=True)
        self.attention = Attention(gru_hidden_size * 2, attention_hidden_size)
        self.linear = nn.Linear(gru_hidden_size * 2, n_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.cnn(x)
        sizes = output.size()
        output = output.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        output = output.transpose(1, 2) # (batch, time, feature)
        output = self.fully_connected(output)
        output, hidden = self.rnn(output)
        output = self.attention(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output

class Food_model(nn.Module):
    hparams = {
        "n_mels": 40,
        "cnn_channels": 16,
        "cnn_kernel_size": (20, 5),
        "stride": (8, 2),
        "gru_hidden_size": 64,
        "attention_hidden_size": 64,
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 50, 
        "test_size": 0.25,
        "n_classes": 14
    }

    def __init__(self, 
                 n_mels = 40,
                 cnn_channels = 16, 
                 cnn_kernel_size = (20, 5),
                 stride = (8, 2),
                 gru_hidden_size = 64, 
                 attention_hidden_size = 64,
                 n_classes = 0):
      
        super().__init__()
        self.cnn = nn.Conv2d(1, cnn_channels, kernel_size=cnn_kernel_size, stride=stride,
                             padding=(cnn_kernel_size[0]//2, cnn_kernel_size[1]//2))
        self.fully_connected = nn.Linear((n_mels//stride[0] + 1)*cnn_channels, cnn_channels)
        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden_size, 
                          bidirectional=True, batch_first=True)
        self.attention = Attention(gru_hidden_size * 2, attention_hidden_size)
        self.linear = nn.Linear(gru_hidden_size * 2, n_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.cnn(x)
        sizes = output.size()
        output = output.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        output = output.transpose(1, 2) # (batch, time, feature)
        output = self.fully_connected(output)
        output, hidden = self.rnn(output)
        output = self.attention(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output