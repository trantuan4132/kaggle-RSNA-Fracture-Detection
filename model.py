from turtle import forward
import torch
import torch.nn as nn
import timm
from itertools import repeat
import torch.nn.functional as F
import numpy as np
from torch import Tensor

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1]) 
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)

class MLPAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, attention_dim=None):
        super(MLPAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        if self.attention_dim is None:
            self.attention_dim = self.hidden_dim
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)
 
    def forward(self, x):
        """
        :param x: seq_len, batch_size, hidden_dim
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        H = torch.tanh(self.proj_w(x)) # (batch_size, seq_len, hidden_dim)
        
        att_scores = torch.softmax(self.proj_v(H),axis=1) # (batch_size, seq_len)
        
        attn_x = (x * att_scores).sum(1) # (batch_size, hidden_dim)
        return attn_x

class RSNAClassifierWithAttention(nn.Module):
    def __init__(self, model_name, pretrained=False, hidden_dim=256, seq_len=24,
                 checkpoint_path='', drop_path_rate=0.0, dropout=0.1):
        super(RSNAClassifierWithAttention, self).__init__()
        self.seq_len = seq_len
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       checkpoint_path=checkpoint_path,
                                       drop_path_rate=drop_path_rate)

        if 'efficientnet' in model_name:
            cnn_feature = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif "res" in model_name:
            cnn_feature = self.model.fc.in_features
            self.model.global_pool = nn.Identity()
            self.model.fc = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool2d(1)
        
        self.spatialdropout = SpatialDropout(dropout)
        self.gru = nn.GRU(cnn_feature, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.mlp_attention_layer = MLPAttentionNetwork(2 * hidden_dim)
        self.logits = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.GRU):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
        
    def forward(self, x): # (B, seq_len, H, W)
        bs = x.size(0) 
        x = x.reshape(bs*self.seq_len, 1, x.size(2), x.size(3)) # (B*seq_len, 1, H, W)
        features = self.model(x)   
        if "res" in self.model_name:                             
            features = self.pooling(features).view(bs*self.seq_len, -1) # (B*seq_len, cnn_feature)
        features = self.spatialdropout(features)                # (B*seq_len, cnn_feature)
        # print(features.shape)
        features = features.reshape(bs, self.seq_len, -1)       # (B, seq_len, cnn_feature)
        features, _ = self.gru(features)                        # (B, seq_len, hidden_dim*2)
        atten_out = self.mlp_attention_layer(features)          # (B, hidden_dim*2)
        pred = self.logits(atten_out)                           # (B, 1)
        pred = pred.view(bs, -1)                                # (B, 1)
        return pred


class RSNAClassifier(nn.Module):
    def __init__(self, model_name, pretrained=False, checkpoint_path='', 
                 in_chans=3, num_classes=1000, drop_path_rate=0.0, use_seq_layer=False):
        """
        Args:
        -----
        model_name: str
            Name of the model to use.
        pretrained: bool
            Whether to load pretrained weights.
        checkpoint_path: str
            Path to model's pretrained weights.
        in_chans: int
            Number of input channels.
        num_classes: int
            Number of output classes.
        drop_path_rate: float
            Drop path rate for the DropPath function.
        """
        super(RSNAClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       checkpoint_path=checkpoint_path,
                                       drop_path_rate=drop_path_rate)
        n_features = self.model.get_classifier().in_features
        self.model.reset_classifier(num_classes=0)
        self.rnn = None
        if use_seq_layer:
            self.rnn = nn.LSTM(n_features, n_features, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(n_features * 2, num_classes)
        else:
            self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        if self.rnn is not None:
            batch_size, seqlen, ch, h, w = x.shape # (2, 32, 3, 512, 512)
            x = x.view(-1, ch, h, w) # (64, 3, 512, 512)
            x = self.model(x) # (64, n_features)
            x = x.view(batch_size, seqlen, -1) # (2, 32, n_features)
            out = self.rnn(x)[0] # (2, 32, n_features * 2)
        else:
            out = self.model(x)
        return self.fc(out)


if __name__ == '__main__':
    model = RSNAClassifier('convnext_tiny', pretrained=False, num_classes=2, 
                           checkpoint_path='convnext_tiny_22k_1k_384_altered.pth')
    # print(model(torch.randn(32, 3, 224, 224)))
    print(model)