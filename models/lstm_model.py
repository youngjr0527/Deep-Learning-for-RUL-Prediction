import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class LSTM(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, hidden_size, num_layers, seq_length=1):
        super(LSTM, self).__init__()
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
        self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: input features, shape (batch_size, seq_length, input_size)
        :return: prediction results
        """
        # 입력 텐서 형태 확인 및 처리
        # 원래 입력 형태: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        # 초기 hidden state와 cell state 초기화
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)  # hidden state
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)  # internal state
        
        # LSTM 층 적용
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        
        # 마지막 hidden state 사용
        hn_last = hn[-1, :, :]  # shape: (batch_size, hidden_size)
        
        # 완전 연결 레이어를 통과
        out = self.relu(self.fc_1(hn_last))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)  # shape: (batch_size, 1)
        
        return out 