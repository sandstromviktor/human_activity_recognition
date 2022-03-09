import torch
import torch.nn as nn
import torch.nn.functional as F


class CONVLSTM(nn.Module):
    def __init__(self, channel=9, hidden=128, num_classes=6):
        super(CONVLSTM, self).__init__()
        act = nn.ReLU
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(channel, 64, kernel_size=5, padding=0, stride=1, dtype = torch.float64),
            nn.BatchNorm1d(64, dtype= torch.float64),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Dropout(p=0.5),
            act(),
            nn.Conv1d(64, 64, kernel_size=5, padding=0, stride=2, dtype = torch.float64),
            nn.BatchNorm1d(64, dtype= torch.float64),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Dropout(p=0.5),
            act())
        
        self.lstm = nn.LSTM(3648, hidden_size=hidden, num_layers=2, batch_first = True, dtype=torch.float64)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden, dtype=torch.float64),
            act(),
            nn.Linear(hidden, num_classes, dtype=torch.float64)
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x, h = self.lstm(x.view(x.shape[0], 1,-1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x