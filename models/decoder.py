import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn import FullAttention, ProbAttention, AttentionLayer

debug=False

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if debug:
            print("input x ", x.shape)

        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        if debug:
            print("down conv", x.shape)

        return x

class DeConvLayer(nn.Module):
    def __init__(self, c_in, c_out =None):
        super(DeConvLayer, self).__init__()
        c_out = c_in if c_out is None else c_out
        self.upConv = nn.ConvTranspose1d(in_channels=c_in,
                                  out_channels=c_out,
                                  kernel_size=3,
                                  stride=3,
                                  padding=2)
        self.norm = nn.BatchNorm1d(c_out)
        self.activation = nn.ELU()
        # self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.upConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        # x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class YformerDecoderLayer(nn.Module):
    def __init__(self,  cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(YformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        # self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, cross_mask =None):

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        x = x + self.dropout(x)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class YformerDecoderLayer_skipless(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(YformerDecoderLayer_skipless, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn


class YformerDecoder_skipless(nn.Module):
    def __init__(self, attn_layers=None, conv_layers=None, norm_layer=None):
        super(YformerDecoder_skipless, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) if attn_layers is not None else None
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.first_conv = DeConvLayer(c_in=512)
        self.first_attn = YformerDecoderLayer(AttentionLayer(FullAttention(False), d_model=512, n_heads=8), d_model =512, d_ff = 2048)
        self.norm = norm_layer

    def forward(self, x_list, fut_x_list, attn_mask=None):
        # x [B, L, D]
        attns = []
        x = x_list.pop(0)
        fut_x = fut_x_list.pop(0)
        x = self.first_attn(x, fut_x, cross_mask=attn_mask)
        x = self.first_conv(x) # upsample to connect with other layers from encoder

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None: 
            x = self.norm(x)
        return x, attns



class YformerDecoder(nn.Module):
    def __init__(self, attn_layers=None, conv_layers=None, norm_layer=None):
        super(YformerDecoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) if attn_layers is not None else None
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.first_conv = DeConvLayer(c_in=512)
        self.first_attn = YformerDecoderLayer(AttentionLayer(FullAttention(False), d_model=512, n_heads=8), d_model =512, d_ff = 2048)
        self.norm = norm_layer

    def forward(self, x_list, fut_x_list, attn_mask=None):
        # x [B, L, D]
        attns = []
        x = x_list.pop(0)
        fut_x = fut_x_list.pop(0)
        x = self.first_attn(x, fut_x, cross_mask=attn_mask)
        x = self.first_conv(x) # upsample to connect with other layers from encoder
        if self.conv_layers is not None:
            if self.attn_layers is not None:
                for cross_x, cross_fut_x, attn_layer, conv_layer in zip(x_list, fut_x_list, self.attn_layers, self.conv_layers):
                    cross = torch.cat((cross_x,cross_fut_x), dim=1)
                    x = attn_layer(x, cross, cross_mask=attn_mask)
                    x = conv_layer(x)
            else:
                # pipeline for only convolution layers
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None: 
            x = self.norm(x)
        return x, attns