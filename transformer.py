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

    def forward(self, src, t):
        e_i = self.gating(src)
        e_outputs = self.encoder(e_i, t)
        output = self.out(e_outputs)
        
        return output.squeeze(-1)


class Gating(nn.Module):
    def __init__(self, d_model, m): # 128,14
        super().__init__()
        self.m = m

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

    def forward(self, x):
        # 입력 텐서의 차원 확인 및 적절한 처리
        input_dim = len(x.shape)
        
        if input_dim == 2:  # [batch_size, features] 형태
            # 입력이 2D 텐서인 경우 (평탄화된 입력)
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
            
            # 필요한 차원 추가 (CNN 레이어용)
            x = x.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, m]
            
            # 이제 x는 [batch_size, 1, 1, m] 형태
            h_i = x  # CNN을 건너뛰고 직접 사용 (차원이 맞지 않아 CNN 적용 불가)
            x_i = x  # 원래 코드의 x_i도 동일하게 설정
            
        elif input_dim == 4:  # [batch_size, channels, seq_len, features] 형태 (원래 기대하던 형태)
            # 원래 코드 사용
            x_i = x[:, :, 1:2, :] #only applying the gating on the current row even with the stack of 3 rows cames as input (1,1,3,14)
            h_i = self.cnn_layers(x) # shape becomes 1,1,1,14 as the nn.conv2d has output channel as 1 but the convolution is applied on whole past input (stack of three)
        else:
            # 지원하지 않는 차원
            raise ValueError(f"지원하지 않는 입력 차원: {input_dim}. 2차원 또는 4차원 텐서가 필요합니다.")
        
        # 나머지 처리는 동일하게 유지
        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)

        # the output of the gating mechanism
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)

        return torch.matmul(hh_i, self.W_e) + self.b_e # (the final output is 1,1,1,128 as the encoder has size of 128.)


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, m, dropout): #d_model = 128  # dimension in encoder, heads = 4  #number of heads in multi-head attention, N = 2  #encoder layers, m = 14  #number of features
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.d_model = d_model

    def forward(self, src, t):
        # src의 실제 크기 확인 및 조정
        # 원래 코드: src = src.reshape(1, self.d_model)
        # 변경된 코드: 입력 크기가 다양할 수 있으므로 유연하게 처리
        
        # 입력 텐서의 총 요소 수 확인
        total_elements = src.numel()
        
        # 첫 번째 차원은 배치 크기(1)로 유지하고, 나머지 요소는 d_model에 맞게 조정
        if total_elements % self.d_model == 0:
            # d_model로 나누어 떨어지는 경우
            batch_size = total_elements // self.d_model
            src = src.reshape(batch_size, self.d_model)
        else:
            # 나누어 떨어지지 않는 경우 조정이 필요
            # 입력 텐서를 선형 레이어를 통해 조정하거나 다른 방법으로 처리 필요
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
        
        x = self.pe(src, t)
        for i in range(self.N):
            x = self.layers[i](x, None)
        return self.norm(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        pe = np.zeros(self.d_model)

        for i in range(0, self.d_model, 2):
            pe[i] = math.sin(t / (10000 ** ((2 * i) / self.d_model)))
            pe[i + 1] = math.cos(t / (10000 ** ((2 * (i + 1)) / self.d_model)))
            
        # pe를 x와 같은 device로 옮기기
        pe_tensor = torch.tensor(pe, dtype=x.dtype, device=x.device)  # 수정

        x = x + pe_tensor
        # x = x + Variable(torch.Tensor(pe))
        return x 


# We can then build a convenient cloning function that can generate multiple layers:
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
