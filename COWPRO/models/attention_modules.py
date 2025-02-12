import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

class RevScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(k.transpose(1, 2), v/self.temperature)
        attn = F.softmax(attn, dim=-1)
        output = q@attn
        return output, attn
    
class RevSelfScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, v, k, mask=None):
        attn = torch.matmul(k.transpose(1, 2), v/self.temperature)
        attn = F.softmax(attn, dim=-1)
        output = v@attn
        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q/self.temperature, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        output = attn@v
        return output, attn
    
class SelfScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, v, k, mask=None):
        attn = torch.matmul(v/self.temperature, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        output = attn@v
        return output, attn
    
class LocationBasedAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, inc, outc, attn_dropout=0.0):
        super().__init__()
        self.temperature = inc**0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.inc = inc
        self.Wd = nn.Linear(inc, outc)

    def forward(self, v, k, mask=None):
        attn = torch.matmul(k.transpose(1, 2), v/self.temperature)
        attn = F.softmax(attn, dim=-1)
        output = self.Wd(attn)
        return output, attn
    

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # print(d_k,d_v)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self._reset_parameters()

    def _reset_parameters(self):
        
        xavier_uniform_(self.w_qs.weight)
        xavier_uniform_(self.w_ks.weight)
        xavier_uniform_(self.w_vs.weight)
        # constant_(self.w_qs.bias,0.)
        # constant_(self.w_ks.bias,0.)
        # constant_(self.w_vs.bias,0.)

        xavier_uniform_(self.fc.weight)
        # constant_(self.fc.bias,0.)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # Transpose for attention dot product: b x n x lq x dv
        #q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q,k,v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)

        # q = self.dropout(q)
        # q += residual
        # q = self.layer_norm(q)

        return q, attn

class QKVAttention(nn.Module):
  def __init__(self,n_head,in_c):
    super().__init__()
    self.attention1 = MultiHeadAttention(n_head,d_model=in_c,d_k=in_c//8,d_v=in_c//8)

    # Two-layer MLP
    self.linear_net = nn.Sequential(
        nn.Linear(in_c, 2*in_c),
        nn.Dropout(0.1),
        nn.ReLU(inplace=True),
        nn.Linear(2*in_c, in_c)
    )

    # Layers to apply in between the main layers
    self.norm1 = nn.LayerNorm(in_c)
    self.norm2 = nn.LayerNorm(in_c)
    self.dropout = nn.Dropout(0.1)

    self.residual = None
  
  def forward(self,q,k,v):
    B,C,H,W = q.shape

    q = q.view(B,C,-1).transpose(1,2)
    k = k.view(B,C,-1).transpose(1,2)
    v = v.view(B,C,-1).transpose(1,2)

    self.residual = v
                           
    qf,_ = self.attention1(q,k,v)

    x = self.residual + self.dropout(qf)
    x = self.norm1(x)

    # MLP part
    linear_out = self.linear_net(x)
    x = x + self.dropout(linear_out)
    qf = self.norm2(x)

    qf = qf.transpose(1,2)
    qf = qf.view(B,C,H,W)
    
    return qf

class VKVAttention(nn.Module):
  def __init__(self,in_c):
    super().__init__()
    self.attention1 = MultiHeadAttention(1,d_model=in_c,d_k=in_c//8,d_v=in_c//8)
  
  def forward(self,v,k):
    B,C,H,W = v.shape

    # q = q.view(B,C,-1).transpose(1,2)
    k = k.view(B,C,-1).transpose(1,2)
    v = v.view(B,C,-1).transpose(1,2)
                           
    qf,_ = self.attention1(v,k,v)
    qf = qf.transpose(1,2)
    qf = qf.view(B,C,H,W)
    
    return qf

