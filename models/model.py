import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack,  YformerEncoder
from models.decoder import Decoder, DecoderLayer, YformerDecoderLayer, YformerDecoder, DeConvLayer, YformerDecoder_skipless, YformerDecoderLayer_skipless
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

debug=False

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, x_list = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

class Yformer_skipless(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(Yformer_skipless, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        # TODO: change the embedding so that there is a simple shared embedding for timestamp 
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.fut_enc_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = YformerEncoder(
            [
                # uses probSparse attention
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Future encoder
        self.future_encoder = YformerEncoder(
            [
                # uses masked attention
                EncoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        # Decoder
        self.udecoder = YformerDecoder_skipless(
            [
                # single attention block in the decoder compared to 2 in the informer
                YformerDecoderLayer_skipless(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(d_layers)
            ],
            [
                DeConvLayer(
                    d_model
                ) for l in range(d_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.seq_len_projection = nn.Linear(d_model, c_out, bias=True) # (bs, 336, 512) -> (bs, 336 + 336, 7) 
        self.pred_len_projection = nn.Linear(d_model, c_out, bias=True) # (bs, 336, 512) -> (bs, 336 + 336, 7) 

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, x_list = self.encoder(enc_out, attn_mask=enc_self_mask)
        x_list.reverse()
        # print("input shape x_dec, x_mark_dec",  x_dec.shape, x_mark_dec.shape)

        # Future Encoder
        fut_enc_out = self.fut_enc_embedding(x_dec, x_mark_dec)
        fut_enc_out, attns, fut_x_list = self.future_encoder(fut_enc_out, attn_mask=enc_self_mask)
        fut_x_list.reverse()

        # Decoder
        dec_out, attns = self.udecoder(x_list, fut_x_list, attn_mask=dec_self_mask)
        # dec_out = self.udecoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        seq_len_dec_out = self.pred_len_projection(dec_out)[:, -(self.seq_len):,:]
        pre_len_dec_out = self.seq_len_projection(dec_out)[:, -(self.pred_len):,:]

        dec_out = torch.cat((seq_len_dec_out, pre_len_dec_out), dim=1)  # 336 -> 336 + 336
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out # [B, L, D]


class Yformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(Yformer, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        # TODO: change the embedding so that there is a simple shared embedding for timestamp 
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.fut_enc_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = YformerEncoder(
            [
                # uses probSparse attention
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Future encoder
        self.future_encoder = YformerEncoder(
            [
                # uses masked attention
                EncoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        # Decoder
        self.udecoder = YformerDecoder(
            [
                # single attention block in the decoder compared to 2 in the informer
                YformerDecoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(d_layers)
            ],
            [
                DeConvLayer(
                    d_model
                ) for l in range(d_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.seq_len_projection = nn.Linear(d_model, c_out, bias=True) # (bs, 336, 512) -> (bs, 336 + 336, 7) 
        self.pred_len_projection = nn.Linear(d_model, c_out, bias=True) # (bs, 336, 512) -> (bs, 336 + 336, 7) 

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, x_list = self.encoder(enc_out, attn_mask=enc_self_mask)
        x_list.reverse()
        # print("input shape x_dec, x_mark_dec",  x_dec.shape, x_mark_dec.shape)

        # Future Encoder
        fut_enc_out = self.fut_enc_embedding(x_dec, x_mark_dec)
        fut_enc_out, attns, fut_x_list = self.future_encoder(fut_enc_out, attn_mask=enc_self_mask)
        fut_x_list.reverse()

        # Decoder
        dec_out, attns = self.udecoder(x_list, fut_x_list, attn_mask=dec_self_mask)

        seq_len_dec_out = self.pred_len_projection(dec_out)[:, -(self.seq_len):,:]
        pre_len_dec_out = self.seq_len_projection(dec_out)[:, -(self.pred_len):,:]

        dec_out = torch.cat((seq_len_dec_out, pre_len_dec_out), dim=1)  # 336 -> 336 + 336
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out # [B, L, D]

class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1)) # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
