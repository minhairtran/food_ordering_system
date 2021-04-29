import sys
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

import torch.nn as nn
import torch.nn.functional as F

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

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 3,
        "rnn_dim": 64,
        "n_feats": 128,
        "dropout": 0.1,
        "stride": 2,
        "learning_rate": 5e-4,
        "batch_size": 4,
        "epochs": 10, 
        "test_size": 0.2
    }

    hparams_confirming = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 2,
        "rnn_dim": 64,
        "n_feats": 128,
        "dropout": 0.1,
        "stride": 2,
        "learning_rate": 5e-4,
        "batch_size": 4,
        "epochs": 30, 
        "test_size": 0.2
    }
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, n_feats):
        super(CNN, self).__init__()

        self.cnn = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel, stride=stride, padding=kernel//2)
        self.layer_norm = CNNLayerNorm(n_feats)
        self.max_pooling = nn.MaxPool2d(kernel, stride, kernel//2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.layer_norm(x)
        x = self.max_pooling(x)
        return x  # (batch, channel, feature, time)

class ConfirmingModel(nn.Module):
    hparams = {
        "n_cnn_layers": 3,
        "dropout": 0.1,
        "stride": 2,
        "learning_rate": 5e-4,
        "batch_size": 4,
        "epochs": 10, 
        "test_size": 0.2,
        "n_feats": 128,
    }

    def __init__(self, n_cnn_layers, n_class, n_feats, stride=2, dropout=0.1):
        super(ConfirmingModel, self).__init__()
        n_feats = n_feats//2

        # n residual cnn layers with filter size of 32
        self.cnn_layers = nn.Sequential(*[
            CNN(in_channels=1 if i==0 else 32, out_channels=32, kernel=3, stride=1, n_feats=n_feats) 
            for i in range(n_cnn_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(n_feats*32, n_feats),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_feats, n_class)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.classifier(x)
        return x