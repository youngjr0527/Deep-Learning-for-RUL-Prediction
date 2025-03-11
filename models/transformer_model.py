import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


'''
num_epochs = 1 # Number of training epochs
d_model = 128  # dimension in encoder
heads = 4  # number of heads in multi-head attention
N = 2  # number of encoder layers
m = 14  # number of features
'''

class Transformer(nn.Module):
    def __init__(self, m, d_model, N, heads, dropout):
        super().__init__()
        # GCU 임베딩 레이어 사용
        self.embedding = GCU(m, d_model)
        self.encoder = Encoder(d_model, N, heads, m, dropout)
        self.out = nn.Linear(d_model, 1)
        self.m = m  # 특성 수 저장
        self.d_model = d_model  # 모델 차원 저장

    def forward(self, src, t):
        # 입력 크기 및 타입 디버깅 (주석 처리)
        # print(f"Transformer 입력 크기: {src.shape}, 타입: {src.dtype}")
        
        # GCU 레이어를 통한 특성 변환
        e_i = self.embedding(src)  # e_i 크기: [batch_size, d_model]
        
        # 인코더 통과
        e_outputs = self.encoder(e_i, t)  # e_outputs 크기: [batch_size, d_model]
        
        # 출력 레이어 통과
        output = self.out(e_outputs)  # output 크기: [batch_size, 1]
        
        # 배치 처리를 위해 return 방식 수정
        return output.squeeze(-1)  # 마지막 차원(크기 1)을 제거


# Gated Convolutional Unit (GCU)
class GCU(nn.Module):
    def __init__(self, m, d_model):
        super().__init__()
        self.m = m
        self.d_model = d_model
        
        # 1D 컨볼루션 레이어 (특성 추출)
        self.feature_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        
        # 게이팅을 위한 컨볼루션 레이어
        self.gate_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 가중치 초기화
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.feature_conv.weight)
        nn.init.xavier_uniform_(self.gate_conv.weight)
        
    def forward(self, x):
        # 입력 텐서의 차원 확인 및 적절한 처리
        input_dim = len(x.shape)
        
        # 4차원 입력 처리 (batch, channel, seq_len, features)
        if input_dim == 4:
            batch_size = x.size(0)
            # 중간 시점(t)의 특성만 사용
            x_i = x[:, :, 1, :]  # [batch_size, channels, features]
            # 차원 재조정
            x_i = x_i.view(batch_size, -1)  # [batch_size, channels*features]
            
            # 특성 수 조정 (필요한 경우)
            if x_i.size(1) != self.m:
                # 선형 보간으로 크기 조정
                x_i = F.interpolate(x_i.unsqueeze(1), size=self.m, mode='linear').squeeze(1)
            
        # 3차원 입력 처리 (batch, seq_len, features)
        elif input_dim == 3:
            batch_size = x.size(0)
            # 마지막 시점의 특성 사용
            x_i = x[:, -1, :]  # [batch_size, features]
            
            # 특성 수 조정 (필요한 경우)
            if x_i.size(1) != self.m:
                # 선형 보간으로 크기 조정
                x_i = F.interpolate(x_i.unsqueeze(1), size=self.m, mode='linear').squeeze(1)
                
        # 2차원 입력 처리 (batch, features)
        elif input_dim == 2:
            x_i = x
            
            # 특성 수 조정 (필요한 경우)
            if x_i.size(1) != self.m:
                # 선형 보간으로 크기 조정
                x_i = F.interpolate(x_i.unsqueeze(1), size=self.m, mode='linear').squeeze(1)
        else:
            # 지원하지 않는 차원
            raise ValueError(f"지원하지 않는 입력 차원: {input_dim}. 2차원, 3차원 또는 4차원 텐서가 필요합니다.")
        
        # 1D 컨볼루션을 위한 차원 변환
        x_conv = x_i.unsqueeze(1)  # [batch_size, 1, features]
        
        # 특성 추출 및 게이팅 적용
        features = self.feature_conv(x_conv)  # [batch_size, d_model, features]
        gates = torch.sigmoid(self.gate_conv(x_conv))  # [batch_size, d_model, features]
        
        # 게이트와 특성 결합 (요소별 곱셈)
        gated_features = features * gates  # [batch_size, d_model, features]
        
        # 전역 평균 풀링으로 시퀀스 차원 제거
        output = F.adaptive_avg_pool1d(gated_features, 1).squeeze(-1)  # [batch_size, d_model]
        
        # 드롭아웃 및 레이어 정규화 적용
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, m, dropout): #d_model = 128  # dimension in encoder, heads = 4  #number of heads in multi-head attention, N = 2  #encoder layers, m = 14  #number of features
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.d_model = d_model
        
        # 추가: 입력 임베딩 처리를 위한 레이어 - 수정: 조건부로 적용하도록 변경
        self.input_embedding = nn.Linear(m, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, t):
        # 유연한 입력 처리
        if isinstance(src, tuple) or isinstance(src, list):
            src = torch.cat([s.view(s.size(0), -1) for s in src], dim=1)
        
        # 입력 차원 확인 및 조정
        if len(src.shape) == 2:  # [batch_size, features]
            # 입력 차원 확인 - 여기서 d_model 크기의 입력이 이미 제공되는 경우 변환하지 않음
            last_dim = src.size(-1)
            
            if last_dim == self.d_model:
                # 이미 Gating 클래스에서 d_model 크기로 변환된 경우 별도 처리 없이 사용
                x = self.dropout(src)
            else:
                # 원래 특성 크기의 입력인 경우 변환
                x = self.input_embedding(src)
                x = self.dropout(x)
            
            # 위치 인코딩 적용
            x = self.pe(x, t)
            
            # 인코더 레이어 적용
            for i in range(self.N):
                x = self.layers[i](x, None)
                
            return self.norm(x)
            
        else:
            # 기존 코드의 접근 방식 유지하되 개선
            # 입력 텐서의 총 요소 수 확인
            total_elements = src.numel()
            
            # 첫 번째 차원은 배치 크기(1)로 유지하고, 나머지 요소는 d_model에 맞게 조정
            if total_elements % self.d_model == 0:
                # d_model로 나누어 떨어지는 경우
                batch_size = total_elements // self.d_model
                src = src.reshape(batch_size, self.d_model)
            else:
                # 나누어 떨어지지 않는 경우 조정이 필요
                # 일단 1차원으로 펼친 다음 필요한 크기로 자르거나 패딩
                src_flat = src.reshape(-1)
                # 선형 보간으로 크기 조정 (간단한 해결책)
                if total_elements > self.d_model:
                    # 다운샘플링 (크기 줄이기)
                    indices = torch.linspace(0, total_elements-1, self.d_model).long()
                    src = src_flat[indices].reshape(1, self.d_model)
                else:
                    # 업샘플링 (크기 늘리기) - 패딩 사용
                    src = torch.zeros(1, self.d_model, device=src.device)
                    src[0, :total_elements] = src_flat
            
            # 임베딩 적용
            src = self.dropout(src)
            
            x = self.pe(src, t)
            for i in range(self.N):
                x = self.layers[i](x, None)
            return self.norm(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 학습 가능한 스케일 파라미터 제거 (원본 코드에 없음)
        # self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        # 원본 코드와 일치하도록 위치 인코딩 수정
        # t는 시퀀스 내 위치(time step)를 나타냄
        batch_size = x.size(0)
        
        # 각 배치 항목에 대한 위치 인코딩 계산
        pe = torch.zeros(batch_size, self.d_model, device=x.device)
        
        # t를 시퀀스 내 위치로 사용
        position = t if isinstance(t, (int, float)) else 0
        
        for i in range(0, self.d_model, 2):
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * 
                               (-math.log(10000.0) / self.d_model))
            
            # 표준 위치 인코딩 공식 사용
            pe[:, i] = torch.sin(position * div_term[i//2])
            if i + 1 < self.d_model:
                pe[:, i + 1] = torch.cos(position * div_term[i//2])
                
        # 위치 인코딩을 추가
        x = x + pe
        
        return x


# We can then build a convenient cloning function that can generate multiple layers:
# 여러 개의 레이어를 생성하는 편리한 함수
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.5):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.5):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
    # scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.5):
        super().__init__()
        # set d_ff as a default to 512
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x 