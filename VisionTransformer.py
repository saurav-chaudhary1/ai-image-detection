import torch.nn as nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self, image_size : tuple[int , int] , embed_dim : int = 768 , patch_size : int = 16 , batch_size : int = 16 , dropout : float = 0.1):
        super().__init__()
        
        num_of_patches = int((image_size[0] * image_size[1])/patch_size**2)
        self.num_of_patches = num_of_patches
        
        
        self.patch_layer = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        ) # patch dimension = (batch , embed_dim , num_patches_H , num_patches_W)
        
        self.class_tokens = nn.Parameter(
            torch.zeros(1 , 1 , embed_dim) , requires_grad=True
        )
        
        self.positional_tokens = nn.Parameter(
            torch.ones(1 , num_of_patches + 1 , embed_dim) , requires_grad=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        nn.init.trunc_normal_(self.class_tokens , std=0.02)
        nn.init.trunc_normal_(self.positional_tokens , std=0.02)
        
    def forward(self , x):
        Batch_size = x.size(0)
        
        patches = self.patch_layer(x)
        flattened_patches = nn.Flatten(start_dim=2 , end_dim=3)(patches).permute(0 , 2 , 1) # -> (batch , embed_dim , num_patches) -> (batch , num_patches , embed_dim)
        
        cls_tokens = self.class_tokens.expand(Batch_size , -1 , -1)
        flattened_patches = torch.concat((cls_tokens , flattened_patches) , dim=1)
        
        positional_tokens = self.positional_tokens.expand(Batch_size , -1 , -1)
        flattened_patches += positional_tokens
        return self.dropout(flattened_patches)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim : int = 768 , h : int = 12 , dropout : float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.h = h
        
        assert embed_dim % h == 0 , "embed_dim should be perfectly divisible by h"
        self.d_k = embed_dim//h
        
        self.w_k = nn.Linear(embed_dim , embed_dim)
        self.w_q = nn.Linear(embed_dim , embed_dim)
        self.w_v = nn.Linear(embed_dim , embed_dim)
        self.w_o = nn.Linear(embed_dim , embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def SelfAttention(key , query , value , mask , dropout):
        d_k = key.shape[-1]
        # (batch , h , seq_len , d_k) @ (batch , h , d_k , seq_len) -> (batch , h , seq_len , d_k)
        attention_scores = (query @ key.transpose(-2 , -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0 , float('-inf'))
        attention_scores = torch.softmax(attention_scores , dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value) , attention_scores
        
    def forward(self , key , query , value , mask = None):
        key = self.w_k(key)
        query = self.w_q(query)
        value = self.w_v(value)
        
        key = key.view(key.shape[0] , key.shape[1] , self.h , self.d_k).permute(0 , 2 , 1 , 3)
        query = query.view(query.shape[0] , query.shape[1] , self.h , self.d_k).permute(0 , 2 , 1 , 3)
        value = value.view(value.shape[0] , value.shape[1] , self.h , self.d_k).permute(0 , 2 , 1 , 3)
        
        x , attention_scores = MultiHeadSelfAttention.SelfAttention(key , query , value , mask ,dropout=self.dropout)
        # x = (batch , h , seq_len , d_k) -> (batch , seq_len , h , d_k) -> (batch , seq_len , embed_dim)
        x = x.permute(0 , 2 , 1 , 3).contiguous().view(x.shape[0] , -1 , self.h * self.d_k)
        return self.w_o(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_dim : int ,hidden_dim : int , dropout : float = 0.1 ):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(in_features=embed_dim , out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim , out_features=embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self , x):
        return self.mlp_layer(x)
    
class ViTBlock(nn.Module):
    def __init__(self, embed_dim : int = 256 , h : int = 8 , hidden_dim : int = 1024 , dropout : float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.MHSA = MultiHeadSelfAttention(embed_dim , h , dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.MLP = MultiLayerPerceptron(embed_dim , hidden_dim , dropout)
        
    def forward(self , x):
        x = x + self.MHSA(self.norm1(x) , self.norm1(x) , self.norm1(x) , None)
        x = x + self.MLP(self.norm2(x))
        return x