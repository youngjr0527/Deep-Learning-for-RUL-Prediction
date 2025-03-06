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
        self.gating = Gating(d_model, m)
        self.encoder = Encoder(d_model, N, heads, m, dropout)
        self.out = nn.Linear(d_model, 1)
        self.m = m  # 특성 수 저장
        self.d_model = d_model  # 모델 차원 저장

    def forward(self, src, t):
        # 입력 크기 및 타입 디버깅 (주석 처리)
        # print(f"Transformer 입력 크기: {src.shape}, 타입: {src.dtype}")
        
        # Gating 메커니즘을 통한 특성 변환
        e_i = self.gating(src)  # e_i 크기: [batch_size, d_model]
        
        # 인코더 통과
        e_outputs = self.encoder(e_i, t)  # e_outputs 크기: [batch_size, d_model]
        
        # 출력 레이어 통과
        output = self.out(e_outputs)  # output 크기: [batch_size, 1]
        
        # 배치 처리를 위해 return 방식 수정
        return output.squeeze(-1)  # 마지막 차원(크기 1)을 제거


class Gating(nn.Module):
    def __init__(self, d_model, m): # 128,14
        super().__init__()
        self.m = m
        self.d_model = d_model

        # 선형 변환 레이어 추가 - 입력 특성을 더 효과적으로 처리
        self.input_projection = nn.Linear(m, d_model)
        
        # 게이팅 메커니즘 개선
        self.gate = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        
        # 피드포워드 네트워크로 변환
        self.ff1 = nn.Linear(d_model, d_model * 2)
        self.ff2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)

        # the reset gate r_i
        self.W_r = nn.Parameter(torch.Tensor(m, m))
        self.V_r = nn.Parameter(torch.Tensor(m, m))
        self.b_r = nn.Parameter(torch.Tensor(m))

        # the update gate u_i
        self.W_u = nn.Parameter(torch.Tensor(m, m))
        self.V_u = nn.Parameter(torch.Tensor(m, m))
        self.b_u = nn.Parameter(torch.Tensor(m))

        # the output
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1), 
        )
        

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.m)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # 추가된 레이어 초기화
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.ff1.weight)
        nn.init.xavier_uniform_(self.ff2.weight)

    def forward(self, x):
        # 입력 텐서의 차원 확인 및 적절한 처리
        input_dim = len(x.shape)
        
        # 원본 코드와 일치하도록 4차원 입력 처리 (batch, channel, seq_len, features)
        if input_dim == 4:  # [batch_size, channels, seq_len, features] 형태
            # 원본 코드 사용
            x_i = x[:, :, 1:2, :]  # 중간 행만 사용 (t 시점)
            h_i = self.cnn_layers(x)  # 3개 행에 대해 컨볼루션 적용 (t-1, t, t+1)
            
            # 게이팅 메커니즘 적용
            r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
            u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)

            # the output of the gating mechanism
            hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)
            
            output = torch.matmul(hh_i, self.W_e) + self.b_e
            
            # 출력 형태 재조정
            output = output.view(-1, self.d_model)
            
            return output
            
        # 3차원 입력 처리 (batch, seq_len, features) - 이 경우 3개의 연속된 시간 단계로 변환
        elif input_dim == 3:
            batch_size = x.size(0)
            seq_len = x.size(1)
            feature_size = x.size(2)
            
            # 시퀀스 길이가 충분한 경우에만 처리
            if seq_len >= 3:
                # 마지막 3개의 시간 단계 선택 (또는 필요에 따라 다른 위치 선택)
                last_steps = x[:, -3:, :]  # (batch_size, 3, features)
                
                # 4차원으로 변환 (batch_size, 1, 3, features)
                x_reshaped = last_steps.unsqueeze(1)
                
                # 4차원 처리 로직 재사용
                x_i = x_reshaped[:, :, 1:2, :]  # 중간 행만 사용
                h_i = self.cnn_layers(x_reshaped)  # 3개 행에 대해 컨볼루션 적용
                
                # 게이팅 메커니즘 적용
                r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
                u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)
                
                # the output of the gating mechanism
                hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)
                
                output = torch.matmul(hh_i, self.W_e) + self.b_e
                
                # 출력 형태 재조정
                output = output.view(batch_size, self.d_model)
                
                return output
            else:
                # 시퀀스 길이가 3보다 작은 경우 패딩 처리
                padded_x = torch.zeros(batch_size, 3, feature_size, device=x.device)
                padded_x[:, -seq_len:, :] = x
                
                # 4차원으로 변환 후 처리
                x_reshaped = padded_x.unsqueeze(1)  # (batch_size, 1, 3, features)
                
                # 이하 동일한 처리
                x_i = x_reshaped[:, :, 1:2, :]
                h_i = self.cnn_layers(x_reshaped)
                
                r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
                u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)
                
                hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)
                
                output = torch.matmul(hh_i, self.W_e) + self.b_e
                output = output.view(batch_size, self.d_model)
                
                return output
                
        # 2차원 입력 처리 (batch, features) - 단일 시간 단계
        elif input_dim == 2:
            batch_size = x.size(0)
            
            # 입력 크기 확인
            if x.size(1) < self.m:
                # 특성 수가 부족한 경우 패딩
                padded_x = torch.zeros(batch_size, self.m, device=x.device)
                padded_x[:, :x.size(1)] = x
                x = padded_x
            elif x.size(1) > self.m:
                # 특성 수가 너무 많은 경우 자르기
                x = x[:, :self.m]
            
            # 단일 시간 단계를 3개로 복제하여 4차원으로 변환
            x_repeated = x.unsqueeze(1).repeat(1, 3, 1)  # (batch_size, 3, features)
            x_reshaped = x_repeated.unsqueeze(1)  # (batch_size, 1, 3, features)
            
            # 이하 동일한 처리
            x_i = x_reshaped[:, :, 1:2, :]
            h_i = self.cnn_layers(x_reshaped)
            
            r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
            u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)
            
            hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)
            
            output = torch.matmul(hh_i, self.W_e) + self.b_e
            output = output.view(batch_size, self.d_model)
            
            return output
            
        else:
            # 지원하지 않는 차원
            raise ValueError(f"지원하지 않는 입력 차원: {input_dim}. 2차원, 3차원 또는 4차원 텐서가 필요합니다.")


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
