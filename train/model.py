import sys
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

import torch
import torch.nn as nn

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
        "cnn_kernel_size": 51,
        "gru_hidden_size": 64,
        "attention_hidden_size": 64,
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 50, 
        "test_size": 0.1,
        "n_classes": 3
    }

    def __init__(self, 
                 n_mels = 40,
                 cnn_channels = 16, 
                 cnn_kernel_size = 51,
                 gru_hidden_size = 64, 
                 attention_hidden_size = 64,
                 n_classes = 0):
      
        super().__init__()
        self.cnn = nn.Conv1d(n_mels, cnn_channels, kernel_size=cnn_kernel_size, 
                             padding=cnn_kernel_size // 2)
        self.relu = nn.ReLU()
        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden_size, 
                          bidirectional=True, batch_first=True)
        self.attention = Attention(gru_hidden_size * 2, attention_hidden_size)
        self.linear = nn.Linear(gru_hidden_size * 2, n_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.cnn(x)
        output = self.relu(output).permute(0, 2, 1)
        output, hidden = self.rnn(output)
        output = self.attention(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output
    

    def inference(self, x, window_size = 100):
        if window_size > x.shape[2]:
            window_size = x.shape[2]
        probabilities = []
        for i in range(window_size, x.shape[2] + 1, 40):
            win = x[:, :, i - window_size:i]
            win = self.cnn(win)
            win = self.relu(win)
            win = win.permute(0, 2, 1)
            win, _ = self.rnn(win)
            win = self.attention(win)
            win = self.linear(win)
            pr = torch.softmax(win, dim=1)
            probabilities.append(pr[0][1].item())
        return probabilities

class Food_model(nn.Module):
    hparams = {
        "n_mels": 40,
        "cnn_channels": 16,
        "cnn_kernel_size": 51,
        "gru_hidden_size": 64,
        "attention_hidden_size": 64,
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 50, 
        "test_size": 0.1,
        "n_classes": 14
    }

    def __init__(self, 
                 n_mels = 40,
                 cnn_channels = 16, 
                 cnn_kernel_size = 51,
                 gru_hidden_size = 64, 
                 attention_hidden_size = 64,
                 n_classes = 0):
      
        super().__init__()
        self.cnn = nn.Conv1d(n_mels, cnn_channels, kernel_size=cnn_kernel_size, 
                             padding=cnn_kernel_size // 2)
        self.relu = nn.ReLU()
        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden_size, 
                          bidirectional=True, batch_first=True)
        self.attention = Attention(gru_hidden_size * 2, attention_hidden_size)
        self.linear = nn.Linear(gru_hidden_size * 2, n_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.cnn(x)
        output = self.relu(output).permute(0, 2, 1)
        output, hidden = self.rnn(output)
        output = self.attention(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output