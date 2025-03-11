import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    """BiLSTM architecture"""

    def __init__(self, input_size):
        super(BiLSTM, self).__init__()

        self.bilstm1 = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, 
                               batch_first=True, dropout=0.1, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1, 
                               batch_first=True, dropout=0.1, bidirectional=True)

        self.fc_1 = nn.Linear(2 * 32, 16)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: input features, shape (batch_size, seq_length, input_size)
        :return: prediction results
        """
        # 입력 텐서 형태: (batch_size, seq_length, input_size)
        # batch_first=True로 설정했으므로 추가 변환 필요 없음
        
        # 첫 번째 BiLSTM
        output, _ = self.bilstm1(x)  # output: (batch_size, seq_length, 2*hidden_size)
        
        # 두 번째 BiLSTM
        output, (hn, cn) = self.bilstm2(output)  # hn: (2*num_layers, batch_size, hidden_size)
        
        # 마지막 순방향 및 역방향 hidden state 추출
        hn_forward = hn[-2, :, :]  # 마지막 순방향 hidden state
        hn_backward = hn[-1, :, :]  # 마지막 역방향 hidden state
        hn_combined = torch.cat((hn_forward, hn_backward), dim=1)  # (batch_size, 2*hidden_size)
        
        # Fully connected layers
        out = self.relu(self.fc_1(hn_combined))
        out = self.dropout(out)
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)  # (batch_size, 1)
        
        return out 