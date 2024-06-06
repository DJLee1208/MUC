import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from layers.SelfAttention_Family import FullAttention, AttentionLayer
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
    

class ParallelLinear(nn.Module): #! 아 이거 Conv1D kernel=1 로 했으면 channel 별로 다른 linear 곱해지는거 알아서 됐을텐데 괜히 짰누
    def __init__(self, input_dim, output_dim, channels=1):
        super(ParallelLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        k = 1 / math.sqrt(input_dim)
        self.weight = nn.Parameter(torch.empty(channels, input_dim, output_dim).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(channels, output_dim).uniform_(-k, k))

    def forward(self, x):
        # x shape: (batch, channels, input_dim)
        # weight shape: (channels, input_dim, output_dim)
        # output shape: (batch, channels, output_dim)
        return torch.einsum('bci,cio->bco', x, self.weight) + self.bias.unsqueeze(0)
    

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
    
class EncoderLayer_NAS(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_NAS, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None): 
        #! Pre-Norm으로 바꿈
        x = self.norm1(x) # x.shape = (batch*n_vars, patch_num = 12, d_model)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x) #! only attention and residual. no activation.

        return x, attn


#####################################################
# MUC Backbone example                              #
#####################################################
class NASCell(nn.Module):
    MACs: int

class Conv1DEmbed(NASCell): #! 모든 channel에 대해  같은 1dConv 인듯
    def __init__(self, configs, first_layer):
        super(Conv1DEmbed, self).__init__()
        self.window_size = configs.seq_len if first_layer else configs.d_model
        self.embed_dim = configs.d_model
        self.channels = configs.enc_in
        self.kernel_size = 4
       
        self.conv = nn.Conv1d(1, self.embed_dim, self.kernel_size, padding=self.kernel_size//2)
        # self.act = nn.GELU()
        self.dropout = nn.Dropout(p=configs.dropout)
        self.MACs = (self.window_size - self.kernel_size + 1) * self.embed_dim * 1 * self.kernel_size * self.channels 
    
    def forward(self, x):
        B,T,C = x.shape        # (batch, input_len, num_variables)
        x = x.permute(0, 2, 1) # B,C,T
        x = self.conv(x.reshape(-1,1,T)) # B*C, 1, T -> B*C, Embed, T
        x = x.mean(dim=-1)     # B*C, Embed  
        x = x.reshape(B,C,-1)  # B,C,Embed
        return self.dropout(x.permute(0, 2, 1)) # B,Embed,C
    

class MLPEmbed(NASCell): # channel 별로 다른 linear
    def __init__(self, configs, first_layer):
        super(MLPEmbed, self).__init__()
        self.window_size = configs.seq_len if first_layer else configs.d_model
        self.embed_dim = configs.d_model
        self.channels = configs.enc_in
        
        self.linear = ParallelLinear(self.window_size, self.embed_dim, self.channels)
        # self.act = nn.GELU() 
        self.dropout = nn.Dropout(p=configs.dropout)
        self.MACs = self.window_size * self.embed_dim * self.channels
    
    def forward(self, x): # (batch, seq_len, num_variables)
        x = x.permute(0, 2, 1) # B,C,T
        x = self.linear(x) # B,C,Embed
        return self.dropout(x.permute(0, 2, 1))

    
class DataEmbedding_inverted(nn.Module): # 모든 chnnel에서 한 개의 linear
    def __init__(self, configs, first_layer):
        super(DataEmbedding_inverted, self).__init__()
        self.window_size = configs.seq_len if first_layer else configs.d_model
        self.embed_dim = configs.d_model
        self.channels = configs.enc_in
        
        self.value_embedding = nn.Linear(self.window_size, self.embed_dim)
        self.dropout = nn.Dropout(p=configs.dropout)
        self.MACs = self.window_size * self.embed_dim * self.channels

    def forward(self, x): # x.shape = (batch, seq_len, n_var)
        x = x.permute(0, 2, 1) # (batch, n_var, seq_len)
        # x: [Batch Variate Time]
        x = self.value_embedding(x)
        # x: [Batch Variate d_model]
        return self.dropout(x.permute(0, 2, 1)) # (batch, d_model, n_var)
    

class ResidualEmbed(NASCell):
    def __init__(self, configs, first_layer):
        super(ResidualEmbed, self).__init__()
        self.MACs = 0

    def forward(self, x):
        return x


class LSTMEmbed(NASCell): # 모든 chnnel에서 한 개의 LSTM
    def __init__(self, configs, first_layer):
        super(LSTMEmbed, self).__init__()
        self.window_size = configs.seq_len if first_layer else configs.d_model
        self.embed_dim = configs.d_model
        self.channels = configs.enc_in
        
        self.lstm = nn.LSTM(self.window_size, self.embed_dim, batch_first=True)
        self.dropout = nn.Dropout(p=configs.dropout)
        self.MACs = 4 * (self.embed_dim * self.window_size + self.embed_dim**2 + self.embed_dim) # rough estimation

    def forward(self, x): # x.shape = (batch, seq_len, num_variables)
        x = x.permute(0, 2, 1) # (batch, num_variables, seq_len)
        embeddings = []
        
        for i in range(x.size(1)):
            _, (h_n, _) = self.lstm(x[:, i:i+1])
            embeddings.append(h_n)
        embeddings = torch.cat(embeddings, dim=1) # (1, batch*n_var, d_model)
        embeddings = embeddings.permute(1, 0, 2).reshape(x.size(0), x.size(1), -1) # (batch, n_var, d_model)

        return self.dropout(embeddings.permute(0, 2, 1)) # (batch, d_model, n_var)
        
    
class TemporalAttentionEmbed(NASCell): #! attention만 있는 버젼
    def __init__(self, configs, first_layer):
        super(TemporalAttentionEmbed, self).__init__()
        patch_len = 16
        stride = 8
        
        self.seq_len = configs.seq_len if first_layer else configs.d_model
        # self.pred_len = configs.pred_len
        padding = stride
        
        self.token_num = int((configs.seq_len - patch_len) / stride + 2) if first_layer \
            else int((configs.d_model - patch_len) / stride + 2)
        
        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = EncoderLayer_NAS(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) 
        
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2) if first_layer \
            else configs.d_model * int((configs.d_model - patch_len) / stride + 2)
        
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.d_model, head_dropout=configs.dropout) # d_model로 Flatten
        self.MACs = (3+2+1) * configs.d_model * configs.d_model * self.token_num #TODO 맞나?
        
    def forward(self, x): # x.shape = (batch, seq_len, num_variables)
        x = x.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x) # x.shape = (batch*n_vars, token_num, d_model)
        
        # Patch Attention
        enc_out, attns = self.encoder(x) # enc_out.shape = (batch*n_vars, token_num, d_model)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # (batch, n_vars, token_num, d_model)
        enc_out = enc_out.permute(0, 1, 3, 2) # (batch, n_vars, d_model, token_num)
        
        # Flatten
        dec_out = self.head(enc_out)  # dec_out.shape = (batch, n_vars, d_model)        
        return dec_out.permute(0, 2, 1) # (batch, d_model, n_vars)
    

class NASGradientSearchEmbedding(nn.Module):
    def __init__(self, configs, first_layer):
        super(NASGradientSearchEmbedding, self).__init__()
        # TODO: Modify the cells to be searched
        self.cells: list[NASCell] = nn.ModuleList([Conv1DEmbed(configs, first_layer), DataEmbedding_inverted(configs, first_layer), MLPEmbed(configs, first_layer), 
                                     TemporalAttentionEmbed(configs, first_layer), LSTMEmbed(configs, first_layer)] if first_layer \
                                         else [Conv1DEmbed(configs, first_layer), DataEmbedding_inverted(configs, first_layer), MLPEmbed(configs, first_layer), 
                                     TemporalAttentionEmbed(configs, first_layer), LSTMEmbed(configs, first_layer), ResidualEmbed(configs, first_layer)])
        # self.weights = nn.Parameter(torch.randn(len(self.cells)))
        self.weights = nn.Parameter(torch.ones(len(self.cells)) / len(self.cells))
        # self.weights = nn.Parameter(torch.zeros(len(self.cells)))
        # k = 1 / math.sqrt(len(self.cells))
        # self.weights = nn.Parameter(torch.empty(len(self.cells)).uniform_(-k, k))
        
    def forward(self, x):
        x = [cell(x) for cell in self.cells]
        x = torch.stack(x, dim=0) # Cell, B, C, Embed
        # weight = self.weights.softmax(dim=0).view(-1,1,1,1) # Cell, 1, 1, 1
        weight = self.weights.view(-1,1,1,1) # Cell, 1, 1, 1
        x = (x * weight).sum(dim=0) # B, C, Embed
        return x, self.weights.tolist()
    
    @property
    def MACs(self) -> torch.Tensor:
        cell_MACs = torch.tensor([cell.MACs for cell in self.cells], dtype=int).to(self.weights.device)
        MACs = (cell_MACs * self.weights.softmax(dim=0)).sum()
        return MACs
    
    @property
    def LASSO(self) -> torch.Tensor:
        return self.weights.abs().sum()


class NAS_series_decomp(nn.Module): #TODO 이거 아직 아무데도 안씀
    def __init__(self, configs):
        kernel_size = configs.moving_avg
        super(NAS_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return x, res, moving_mean

#############################
# MUC Backbone example      #
#############################
# if __name__ == 'main':
#     from torch.optim import Adam
#     B, Window, Chan, Embed = 32, 96, 10, 512
#     embedding_layer = NASGradientSearchEmbedding(Window, Embed, Chan)
#     model = nn.Linear(Embed, 1) # Dummy model
#     optim = Adam([
#         {'params': model.parameters()},
#         {'params': embedding_layer.parameters(), 'lr': 1e-3},
#     ])
    
#     data = torch.randn(B, Chan, Window)
#     y_hat = torch.randn(B, 1)
    
#     embedding = embedding_layer(data)
#     y = model(embedding)
    
    
#     #TODO 여기 loss 보고 trainer 수정하기
#     loss = F.mse_loss(y, y_hat)
    
#     hparam = {'MACs_weight': 1e-3, 'LASSO_weight': 1e-3}
#     loss = loss + hparam['MACs_weight'] * embedding_layer.MACs() + hparam['LASSO_weight'] * embedding_layer.LASSO()
    
#     optim.zero_grad(set_to_none=True)
#     loss.backward()
#     optim.step()