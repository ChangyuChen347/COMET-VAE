### TAKEN FROM https://github.com/kolloldas/torchnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle as pkl
from model.common_layer import EncoderLayer, DecoderLayer, DecoderLayer_2, VarDecoderLayer, MultiHeadAttention, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask, _get_attn_self_mask,  get_input_from_batch, get_output_from_batch, gaussian_kld,SoftmaxOutputLayer
from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if(self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size,
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size,
                 num_heads,
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout,
                 attention_dropout,
                 relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if(config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        #Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if(self.universal):
            if(config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=200, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if(self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size,
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size,
                 num_heads,
                 _gen_bias_mask(max_length), # mandatory
                 None,
                 layer_dropout,
                 attention_dropout,
                 relu_dropout)

        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        #dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if(self.universal):
            if(config.act):
                x, attn_dist, (self.remainders,self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Decoder_2(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=200, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder_2, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        if (self.universal):
            self.dec = DecoderLayer_2(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer_2(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_src_c, mask_src_r, mask_trg = mask
        # dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, mask_src_c, mask_src_r, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x,  z=None, emo=None):

       
        logit = self.proj(x)       #bs, l, v

       
        return F.log_softmax(logit, dim=-1)


class CodeBook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(CodeBook, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._commitment_cost = commitment_cost
        self._embedding.weight.data.normal_(mean=0, std=0.1)

    def forward(self, c_input, predicted_label=None):
        # Calculate distances
        inputs = c_input # bs * atomic_rel, dim
        #print(inputs.shape)
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).cuda()
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)

        # convert quantized from BHWC -> BCHW
        if config.use_prior or config.use_prior_joint:
            z = self._embedding(predicted_label)
            return loss, z, encodings
        return loss, quantized, encodings






class Latent_(nn.Module):
    def __init__(self):
        super(Latent_, self).__init__()
        self.mean = PositionwiseFeedForward(config.hidden_dim, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.var = PositionwiseFeedForward(config.hidden_dim, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.mean_p = PositionwiseFeedForward(config.hidden_dim*2, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.var_p = PositionwiseFeedForward(config.hidden_dim*2, config.filter, config.hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)

    def forward(self,x,x_p, train=True):
        mean = self.mean(x)
        log_var = self.var(x)
        eps = torch.randn(x.size())
        std = torch.exp(0.5 * log_var)
        if config.USE_CUDA: eps = eps.cuda()
        z = eps * std + mean
        kld_loss = torch.zeros(x.shape[0], 1).cuda()
        if x_p is not None:
            mean_p = self.mean_p(torch.cat((x_p,x),dim=-1))
            log_var_p = self.var_p(torch.cat((x_p,x),dim=-1))
            kld_loss = gaussian_kld(mean_p,log_var_p,mean,log_var)
            kld_loss = torch.mean(kld_loss)
        if train and x_p is not None:
            std = torch.exp(0.5 * log_var_p)
            if config.USE_CUDA: eps = eps.cuda()
            z = eps * std + mean_p
        return kld_loss, z




class CvaeTrans(nn.Module):
    def __init__(self, vocab, emo_number,  model_file_path=None, is_eval=False, load_optim=False, atomic_rel_num=None):
        super(CvaeTrans, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        if config.sep_code:
            self.codebook = nn.ModuleList(
                [CodeBook(config.sep_dim, config.hidden_dim, config.control) for _ in range(atomic_rel_num)])
            self.code_linear = nn.ModuleList(
                [nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(atomic_rel_num)])

        else:
            self.codebook = CodeBook(config.sep_dim, config.hidden_dim, config.control)
       
        self.sent_linear = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.rel_t_emb = nn.Embedding(60, config.hidden_dim)
        self.rel_emb = nn.Embedding(60, config.hidden_dim)
       
        self.choose_z_linear = nn.Linear(2 * config.hidden_dim, 1)
        self.choose_c_linear = nn.Linear(2 * config.hidden_dim, 1)
        self.choose_emo_linear = nn.Linear(2 * config.hidden_dim, 1)
     
        self.embedding = share_embedding(self.vocab,config.pretrain_emb)


      

        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)

        self.atomic_know_encoder = self.encoder
      

        if config.cross:
            self.decoder = Decoder_2(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads,
                                total_key_depth=config.depth,total_value_depth=config.depth,
                                filter_size=config.filter)
        else:
            self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                                     num_heads=config.heads,
                                     total_key_depth=config.depth, total_value_depth=config.depth,
                                     filter_size=config.filter)
        self.linear_drop = nn.Dropout(p=config.linear_dropout)
        self.attn_dropout = nn.Dropout(p=config.attn_dropout)
        self.emo_drop = nn.Dropout(p=0.2)
        if config.train_prior or config.train_prior_joint or config.use_prior_joint:
            self.prior_cls = nn.ModuleList([SoftmaxOutputLayer(config.hidden_dim, config.sep_dim) for _ in range(atomic_rel_num)])
            self.cls_criterion = nn.NLLLoss()
            if config.train_prior_joint or config.use_prior_joint:
                self.prior_cls = nn.ModuleList(
                    [SoftmaxOutputLayer(config.hidden_dim*2, config.sep_dim) for _ in range(atomic_rel_num)])

            #self.c_r_prior_cls = SoftmaxOutputLayer(config.hidden_dim, 300)

        self.atomic_rel_num = atomic_rel_num
        self.codebook_ln = LayerNorm(config.hidden_dim)
        self.ln = LayerNorm(config.hidden_dim)
        self.mm_ln = LayerNorm(config.hidden_dim)
        self.z_ln = LayerNorm(config.hidden_dim)

        self.attnW_add_c = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_add_c = nn.Linear(config.emb_dim, 1, bias=False)
        self.attnW_add_z = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_add_z = nn.Linear(config.emb_dim, 1, bias=False)
        
      
        self.attnW_add_c = nn.ModuleList([nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.attnV_add_c = nn.ModuleList([nn.Linear(config.emb_dim, config.hidden_dim) for _ in range(config.mem_hop)])
        self.attnW_add_z = nn.ModuleList([nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.attnV_add_z = nn.ModuleList([nn.Linear(config.emb_dim, config.hidden_dim) for _ in range(config.mem_hop)])

        self.query_linear = nn.ModuleList([nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.z_anw_linear = nn.ModuleList([nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.c_anw_linear = nn.ModuleList([nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])

        if config.softmax:
            cato = 3
        else:
            cato = 1
        self.gate_linear = nn.ModuleList([nn.Linear(3*config.emb_dim, cato) for _ in range(config.mem_hop)])

        self.attnW_emo_c = nn.ModuleList(
            [nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.attnV_emo_c = nn.ModuleList([nn.Linear(config.emb_dim, config.hidden_dim) for _ in range(config.mem_hop)])

        self.attnW_emo_z = nn.ModuleList(
            [nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.attnV_emo_z = nn.ModuleList([nn.Linear(config.emb_dim, config.hidden_dim) for _ in range(config.mem_hop)])

        self.query_emo_linear = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.z_anw_emo_linear = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.c_anw_emo_linear = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.gate_emo_linear = nn.ModuleList([nn.Linear(3 * config.emb_dim, cato) for _ in range(config.mem_hop)])

        self.attnW_rec_c = nn.ModuleList(
            [nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.attnV_rec_c = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.hidden_dim) for _ in range(config.mem_hop)])

        self.attnW_rec_z = nn.ModuleList(
            [nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.attnV_rec_z = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.hidden_dim) for _ in range(config.mem_hop)])

        self.query_rec_linear = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.z_anw_rec_linear = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.c_anw_rec_linear = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.gate_rec_linear = nn.ModuleList([nn.Linear(3 * config.emb_dim, cato) for _ in range(config.mem_hop)])


        self.attnW_emo = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_emo = nn.Linear(config.emb_dim, 1, bias=False)
        self.attnW_copy_c = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_copy_c = nn.Linear(config.emb_dim, 1, bias=False)
        self.attnW_copy_z = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_copy_z = nn.Linear(config.emb_dim, 1, bias=False)

        self.attnW_emo = nn.ModuleList([nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])
        self.attnV_emo = nn.ModuleList([nn.Linear(config.emb_dim, 1) for _ in range(config.mem_hop)])
        self.emo_linear = nn.ModuleList([nn.Linear(2 * config.emb_dim, config.emb_dim) for _ in range(config.mem_hop)])

        self.attnW_ck = nn.Linear(2 * config.emb_dim, config.emb_dim)
       
        self.attnV_ck = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.attnW_r = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_r = nn.Linear(config.emb_dim, 1, bias=False)
        self.attnW_pointer = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_pointer = nn.Linear(config.emb_dim, 1, bias=False)
        self.attnW_c = nn.Linear(2 * config.emb_dim, config.emb_dim)
        self.attnV_c = nn.Linear(config.emb_dim, 1, bias=False)
        if config.c_r_codebook:
            self.latent = Latent_()

        self.bow = SoftmaxOutputLayer(config.hidden_dim,self.vocab_size)
        self.bow2 = SoftmaxOutputLayer(config.hidden_dim, self.vocab_size)
        if config.c_r_codebook:
            self.bow3 = SoftmaxOutputLayer(config.hidden_dim, self.vocab_size)
        self.atomic_know_cls = SoftmaxOutputLayer(config.hidden_dim, atomic_rel_num+1)
        self.atomic_know_criterion = nn.NLLLoss(ignore_index=0)
        
        kk = 1
        
        sig = 2
        
        self.add_z_linear = nn.Linear(config.hidden_dim*(sig+kk), config.hidden_dim)
        
        
        if config.cat_r:
            self.codebook_linear = nn.Linear(config.hidden_dim * 5, config.hidden_dim)
        else:
            self.codebook_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        
        if config.use_prior_dis:
            self.cls_linear = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        if config.multitask:
            self.emo = SoftmaxOutputLayer(config.hidden_dim,emo_number)
            self.emotion_embedding = nn.Linear(emo_number, config.hidden_dim, bias=False)  #todo update bias
            self.emo_criterion = nn.NLLLoss()
       
        self.generator = Generator(config.hidden_dim, self.vocab_size)
        
        
        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.cuda()
        emo_para = []
        other_para = []
        for name, param in self.named_parameters():
            if name.find('emo') != -1:
                emo_para += [param]
            else:
                other_para += [param]

        
        self.optimizer = torch.optim.Adam([{'params': emo_para, 'betas': (0.9, config.beta2)}, {'params': other_para,
                                                                                        'betas': (0.9, 0.999)}, ],
                                          lr=config.lr, betas=(0.9, config.beta2), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=config.factor,
                                                                    min_lr=1e-6,
                                                                    patience=0, verbose=True, threshold=0.0001,
                                                                    eps=1e-8)

        if (config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000,
                                     torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer.optimizer, mode='min', factor=config.factor, min_lr=1e-6,
                                                                                                        patience=0, verbose = True, threshold = 0.0001, eps = 1e-8)


   

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature=1):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = torch.eq(y, torch.max(y, 1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def train_one_batch(self, batch, iter, train=True):
        return_dict = {}
        atomic_know = batch['atomic_know']
        batchsize, atomic_know_num, atomic_know_len = atomic_know.shape
        flat_atomic_know = atomic_know.reshape(batchsize * atomic_know_num, atomic_know_len)
        mask_atomic_know = flat_atomic_know.data.eq(config.PAD_idx).unsqueeze(1)

        atomic_rel_emb = self.rel_emb(batch["atomic_relation"].repeat(1, 1, atomic_know_len)).reshape(batchsize * atomic_know_num, atomic_know_len, -1)

        atomic_rel_emb_c = self.rel_t_emb(batch['atomic_relation']).squeeze(2)
        atomic_rel_emb_r = self.rel_t_emb(batch["atomic_relation_t"]).squeeze(2)

        atomic_rel_emb_mask = batch['atomic_relation'].data.eq(-1)
        atomic_know_hidden = self.atomic_know_encoder(self.embedding(flat_atomic_know) + atomic_rel_emb,
                                                      mask_atomic_know)
        atomic_know_rel_hidden = (atomic_know_hidden.reshape(batchsize, atomic_know_num, atomic_know_len, -1))[:, :, 0,
                                 :]
       

        atomic_know_flat = batch['atomic_know_t_set_all']

        atomic_know_t = batch['atomic_know_t']
        atomic_rel_emb_t_mask = batch['atomic_relation_t'].data.eq(0)

        _, atomic_know_t_num, atomic_know_t_len = atomic_know_t.shape
        flat_atomic_know_t = atomic_know_t.reshape(batchsize * atomic_know_t_num, atomic_know_t_len)
        mask_atomic_know_t = flat_atomic_know_t.data.eq(config.PAD_idx).unsqueeze(1)
        atomic_rel_t_emb = self.rel_emb(batch["atomic_relation_t"].repeat(1, 1, atomic_know_t_len)).reshape(batchsize * atomic_know_t_num, atomic_know_t_len, -1)
        #todo share seg emb

        atomic_know_t_hidden = self.atomic_know_encoder(self.embedding(flat_atomic_know_t) + atomic_rel_t_emb,
                                                        mask_atomic_know_t)
        atomic_know_t_rel_hidden = atomic_know_t_hidden.reshape(batchsize, atomic_know_t_num, atomic_know_t_len, -1)[:,
                                   :, 0, :]
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        enc_batch_extend_vocab = enc_batch

        posterior_batch = batch["posterior_batch"]
        posterior_batch_emb = self.embedding(posterior_batch)
        mask_res = batch["posterior_batch"].data.eq(config.PAD_idx).unsqueeze(1)
        posterior_mask = self.embedding(batch["posterior_mask"])

        r_encoder_outputs = self.encoder(posterior_batch_emb+posterior_mask,mask_res)
        ## Encode
        mask_src_r = posterior_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        enc_batch_emb = self.embedding(enc_batch)
        encoder_outputs = self.encoder(enc_batch_emb+emb_mask,mask_src)
        # Decode
        c_emb = encoder_outputs[:, 0]
        r_emb = r_encoder_outputs[:, 0]
        #print(batch['atomic_know_t_bow_target_by_rel'].shape)
        _, tgt_len, bow_target_by_rel_len = batch['atomic_know_t_bow_target_by_rel'].shape

        def attn(X, Y, X_mask, Y_mask, attnv, attnw, temp=None):
            # X bs, src_len
            # Y bs, tgt_len
            src_len = X.shape[1]
            tgt_len = Y.shape[1]
            tgt_outputs = Y.repeat(1, 1, src_len).view(batchsize, src_len * tgt_len, config.hidden_dim)
            src_outputs = X.repeat(1, tgt_len, 1).view(batchsize, tgt_len * src_len, config.hidden_dim)
            #  bs, 300
            attn_dist = attnv(
                torch.tanh(attnw(torch.cat([tgt_outputs, src_outputs], dim=-1)))).view(batchsize,
                                                                                                    tgt_len,
                                                                                                    src_len)

            src2tgt = F.softmax(attn_dist + ((X_mask.squeeze(2).unsqueeze(1).repeat(1, tgt_len, 1)) * -10000000000), dim=2)
            return src2tgt

        def attn_r(X, Y, X_mask, Y_mask, attnv, attnw, r_emb, temp=None, hard=True):
            # X bs, src_len
            # Y bs, tgt_len
            src_len = X.shape[1]
            tgt_len = Y.shape[1]
            # print(Y.shape)
            # print(X.shape)
            # print(X_mask.shape)
            tgt_outputs = Y.repeat(1, 1, src_len).view(batchsize, src_len * tgt_len, config.hidden_dim)
            src_outputs = X.repeat(1, tgt_len, 1).view(batchsize, tgt_len * src_len, config.hidden_dim)
            #  bs, 300
            r_emb = r_emb.repeat(1, tgt_len, 1).view(batchsize, tgt_len * src_len, config.hidden_dim)
            b = torch.tanh(attnw(self.linear_drop(torch.cat([tgt_outputs, src_outputs], dim=-1))))  #bs, t*s, dim
            a = attnv(self.linear_drop(r_emb))     #bs, t*s, dim
            #print(attnv.weight.shape) #dim*dim
            if config.dot:
                a = a.reshape(batchsize * tgt_len * src_len, 1, -1)
                b = b.reshape(batchsize * tgt_len * src_len, -1, 1)
                attn_dist = torch.bmm(a, b).view(batchsize,
                                                 tgt_len,
                                                 src_len)
            else:
                a = a.reshape(batchsize * tgt_len * src_len, -1)
                b = b.reshape(batchsize * tgt_len * src_len, -1)
                attn_dist = torch.cosine_similarity(a, b, dim=-1)

                attn_dist = attn_dist.view(batchsize,
                                                 tgt_len,
                                                 src_len)
            if temp is None:
                attn_dist = attn_dist * config.temp
            else:
                attn_dist = attn_dist * temp
            attn_dist = self.attn_dropout(attn_dist)
            #attn_dist = attn_dist * 0.75
            # print('----')
            #print(attn_dist[0,:,:])
            # print(attn_dist.shape, attn_dist)
            #print(F.softmax(attn_dist, dim=2)[0,:,:])
            attn_dist = attn_dist + ((X_mask.squeeze(2).unsqueeze(1).repeat(1, tgt_len, 1)) * -10000000000)

            if hard and config.hard:
                attn_dist = attn_dist.reshape(-1, src_len)
                src2tgt = self.gumbel_softmax(attn_dist, 1, True)
                src2tgt = src2tgt.reshape(batchsize, tgt_len, src_len)
            else:
                src2tgt = F.softmax(attn_dist, dim=2)
            #src2tgt = self.attn_dropout(src2tgt)
            # print(src2tgt.shape, src2tgt)
            #print('----')
            #src2tgt = F.softmax(attn_dist)
            return src2tgt
        query_encoder_outputs = self.encoder(self.embedding(batch["sess"]) + self.embedding(batch["sess_mask"]),
                                             batch["sess"].data.eq(config.PAD_idx).unsqueeze(1))

        q_emb = query_encoder_outputs[:,  0]
        #c_emb += q_emb


        
        rel_target_post_emb = atomic_know_t_rel_hidden

        atomic_rel_emb_mask = atomic_rel_emb_mask.type(
            'torch.FloatTensor').cuda()
        attent_src = atomic_know_rel_hidden
        mask_src_2 = atomic_rel_emb_mask.type('torch.FloatTensor').cuda()
        src2tgt = attn_r(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_ck,
                       attnw=self.attnW_ck, r_emb=atomic_rel_emb_c, temp=config.ck_temp)
        res_tmp = src2tgt.unsqueeze(3) * (attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_ck_post = res_tmp.sum(2)  # bs, tgt_len, dim
        attent_src = encoder_outputs
        mask_src_2 = enc_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_c,
                       attnw=self.attnW_c, temp=config.ck_temp)
        res_tmp = src2tgt.unsqueeze(3) * (attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_c_post = res_tmp.sum(2)  # bs, tgt_len, dim

        attent_src = r_encoder_outputs
        mask_src_2 = posterior_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_r,
                       attnw=self.attnW_r, temp=config.ck_temp)
        res_tmp = src2tgt.unsqueeze(3) * (attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_r_post = res_tmp.sum(2)  # bs, tgt_len, dim

        res_tmp_post = torch.cat((res_tmp_ck_post, res_tmp_c_post, res_tmp_r_post), -1)
        '''

        '''
        rel_target_post_emb = self.rel_t_emb(batch["atomic_relation_t"].reshape(batchsize, self.atomic_rel_num))
        attent_src = atomic_know_rel_hidden
        mask_src_2 = atomic_rel_emb_mask.type('torch.FloatTensor').cuda()
        src2tgt = attn_r(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_ck,
                       attnw=self.attnW_ck, r_emb=atomic_rel_emb_c, temp=config.ck_temp, hard=False)
        res_tmp = src2tgt.unsqueeze(3) * (attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_ck = res_tmp.sum(2)  # bs, tgt_len, dim

        attent_src = encoder_outputs
        mask_src_2 = enc_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_c,
                       attnw=self.attnW_c)
        res_tmp = src2tgt.unsqueeze(3) * (attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_c = res_tmp.sum(2)  # bs, tgt_len, dim

        attent_src = r_encoder_outputs
        mask_src_2 = posterior_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_r,
                       attnw=self.attnW_r)
        res_tmp = src2tgt.unsqueeze(3) * (attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_r = res_tmp.sum(2)  # bs, tgt_len, dim

        res_tmp_pri = torch.cat((res_tmp_ck, res_tmp_c, res_tmp_r), -1)
        res_tmp = res_tmp_post + res_tmp_pri
        if config.cat_r:
            res_tmp = torch.cat((res_tmp, r_emb.unsqueeze(1).repeat(1, tgt_len, 1)), -1)
      
        res_tmp = torch.cat((res_tmp, atomic_know_t_rel_hidden), -1)
        res_tmp = self.codebook_linear(self.linear_drop(res_tmp))
        res_tmp_cls = torch.cat((res_tmp_ck, res_tmp_c), -1)

        if not config.sep_code:
            res_tmp = res_tmp.reshape(batchsize * tgt_len, -1)
        if config.use_prior:
            pred = batch['pri_label'] # bs, rel, 1
            # if config.sep_code:
            #     pred = pred.reshape(batchsize, tgt_len, 1).transpost(0, 1)
        else:
            pred = None



        if config.c_r_codebook:
            if config.use_prior_joint:
                dis_c_r_loss, z_c_r = self.latent(c_emb, None)
            else:
                dis_c_r_loss, z_c_r = self.latent(c_emb, r_emb)
            z_logit = self.bow3(z_c_r)  # [batch_size, vocab_size] #todo z+meta
            z_logit = z_logit.reshape(batchsize, 1, self.vocab_size).repeat(1, dec_batch.shape[1], 1)
            loss_c_r_aux = self.criterion(z_logit.contiguous().view(-1, z_logit.size(-1)),
                                          dec_batch.contiguous().view(-1))

        if config.use_prior_joint:
            clss = []
            for c in range(self.atomic_rel_num):
                clss.append(
                    self.prior_cls[c](res_tmp_cls.reshape(batchsize, tgt_len, -1)[:, c, :]).reshape(batchsize, 1,
                                                                                                    -1))  # 8 * 1 * 400
            cls = torch.cat(clss, 1).view(batchsize * self.atomic_rel_num, -1)

            pred_cls = np.argmax(cls.detach().cpu().numpy(), axis=1)

            #return_dict['encoding'] = pred_cls
            pred = torch.tensor(pred_cls).cuda().reshape(batchsize, tgt_len, -1)

        if config.sep_code:
            dis_all_loss = None
            zs = None
            encodeings = None
            for i in range(self.atomic_rel_num):
                if config.use_prior or config.use_prior_joint:
                    dis_loss, z, encoding = self.codebook[i](res_tmp[:, i, :], predicted_label=pred[:,i,:])
                else:
                    dis_loss, z, encoding = self.codebook[i](res_tmp[:, i, :], None)
                if dis_all_loss is None:
                    dis_all_loss = dis_loss
                    encodeings = encoding.unsqueeze(1)  # bs, 1, 40
                    zs = z.unsqueeze(1)  # bs, 1, dim
                else:
                    zs = torch.cat((zs, z.unsqueeze(1)), 1)
                    encodeings = torch.cat((encodeings, encoding.unsqueeze(1)), 1)
                    dis_all_loss += dis_loss

            encoding = encodeings.reshape(batchsize * tgt_len, -1)
            dis_loss = dis_all_loss
            z = zs.reshape(batchsize, tgt_len, -1)
        else:
            if config.use_prior or config.use_prior_joint:
                dis_loss, z, encoding = self.codebook(res_tmp, predicted_label=pred.reshape(batchsize*tgt_len, -1))
            else:
                dis_loss, z, encoding = self.codebook(res_tmp)

            z = z.reshape(batchsize, tgt_len, -1)
        return_dict['z'] = z.detach().cpu().numpy()
       

        batch_vq_label = encoding
        batch_vq_label = batch_vq_label.detach().cpu().numpy().tolist()
        target = torch.tensor([onehot.index(1) for onehot in batch_vq_label]).cuda()

        #print(pred, target)

        if config.use_prior_dis:
            cls_loss = torch.mean((res_tmp_cls - res_tmp) ** 2)
            return_dict['cls_loss'] = cls_loss.item()

        if config.train_prior_joint:


            clss = []


            for c in range(self.atomic_rel_num):
                                                                        
                clss.append(
                    self.prior_cls[c](res_tmp_cls.reshape(batchsize, tgt_len, -1)[:, c, :]).reshape(batchsize, 1,
                                                                                                    -1))  # 8 * 1 * 400
            cls = torch.cat(clss, 1).view(batchsize * self.atomic_rel_num, -1)
           
            target = target.view(batchsize, self.atomic_rel_num, -1)
            cls_loss = self.cls_criterion(cls, target.view(-1))
            pred_cls = np.argmax(cls.clone().detach().cpu().numpy(), axis=1)
            cls_acc = accuracy_score((target.view(-1)).cpu().numpy(), pred_cls)
            return_dict['c_r_acc'] = 0
            return_dict['loss_vae_cls'] = cls_loss.item()
            return_dict['vae_acc'] = cls_acc
            #return_dict['encoding'] = pred_cls


        res_tmp_logit = self.bow2(z) #bs, tgt_len, vocab

        res_tmp_logit = res_tmp_logit.reshape(batchsize, tgt_len, 1, self.vocab_size).repeat(1, 1, bow_target_by_rel_len, 1)

        loss_aux = self.criterion(res_tmp_logit.view(-1, res_tmp_logit.size(-1)), batch['atomic_know_t_bow_target_by_rel'].view(-1))



       
        sos_token = torch.LongTensor([config.SOS_idx] * batchsize).unsqueeze(1).cuda()
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        input_vector = self.embedding(dec_batch_shift)

        if config.rec_weight != 0 and config.multitask:
            add_z = c_emb
            for i in range(config.mem_hop):
                a2src = attn_r(atomic_know_rel_hidden, add_z.unsqueeze(1), atomic_rel_emb_mask, None,
                               self.attnV_emo_c[i],
                               self.attnW_emo_c[i], atomic_rel_emb_c, temp=config.ck_temp, hard=False)
                # if random.random() < 0.1:
                #     print('emoc', a2src.reshape(batchsize, atomic_know_num, 1)[0,:,:])
                mem_c = (a2src.reshape(batchsize, atomic_know_num, 1) * atomic_know_rel_hidden).sum(1)

                a2src = attn_r(z, add_z.unsqueeze(1), atomic_rel_emb_mask, None, self.attnV_emo_z[i],
                               self.attnW_emo_z[i], atomic_rel_emb_r)
                # if random.random() < 0.1:
                #     print('emoz', a2src.reshape(batchsize, atomic_know_num, 1)[0,:,:])
                
                mem_z = (a2src.reshape(batchsize, atomic_know_num, 1) * z).sum(1)
                beta = self.gate_emo_linear[i](self.linear_drop(torch.cat((add_z, mem_c, mem_z), -1)))
                if config.softmax:
                    beta = F.softmax(beta, -1)
                    emo_z = beta[:, 0:1] * self.c_anw_emo_linear[i](self.linear_drop(mem_c)) + \
                            beta[:, 1:2] * self.z_anw_emo_linear[i](self.linear_drop(mem_z)) + \
                            beta[:, 2:3] * self.query_emo_linear[i](self.linear_drop(add_z))
                else:
                    beta = torch.sigmoid(beta)
                    emo_z = beta * self.c_anw_emo_linear[i](self.linear_drop(mem_c)) + \
                            (1 - beta) * self.z_anw_emo_linear[i](self.linear_drop(mem_z)) + \
                            self.query_emo_linear[i](self.linear_drop(add_z))
                if config.mem_hop > 1:
                    add_z = emo_z
               
            add_z = c_emb
            for i in range(config.mem_hop):
                rand_atomic_rel_emb_mask = torch.randn(atomic_rel_emb_mask.shape,
                                                       device=atomic_rel_emb_mask.device) < config.know_drop
                a2src = attn_r(atomic_know_rel_hidden, add_z.unsqueeze(1), rand_atomic_rel_emb_mask, None,
                               self.attnV_rec_c[i],
                               self.attnW_rec_c[i], atomic_rel_emb_c, temp=config.ck_temp, hard=False)
                # if random.random() < 0.1:
                #     print('recc', a2src.reshape(batchsize, atomic_know_num, 1)[0,:,:])
                rand_atomic_rel_emb_mask = torch.randn(atomic_rel_emb_mask.shape,
                                                       device=atomic_rel_emb_mask.device) < config.know_drop
                mem_c = (a2src.reshape(batchsize, atomic_know_num, 1) * atomic_know_rel_hidden).sum(1)
                a2src = attn_r(z, add_z.unsqueeze(1), rand_atomic_rel_emb_mask, None, self.attnV_rec_z[i],
                               self.attnW_rec_z[i], atomic_rel_emb_r)
                # if random.random() < 0.1:
                #     print('recz', a2src.reshape(batchsize, atomic_know_num, 1)[0,:,:])
                
                mem_z = (a2src.reshape(batchsize, atomic_know_num, 1) * z).sum(1)
                beta = self.gate_rec_linear[i](self.linear_drop(torch.cat((add_z, mem_c, mem_z), -1)))
                if config.softmax:
                    beta = F.softmax(beta, -1)
                    rec_z = beta[:, 0:1] * self.c_anw_rec_linear[i](self.linear_drop(mem_c)) + \
                            beta[:, 1:2] * self.z_anw_rec_linear[i](self.linear_drop(mem_z)) + \
                            beta[:, 2:3] * self.query_rec_linear[i](self.linear_drop(add_z))
                else:
                    beta = torch.sigmoid(beta)
                    rec_z = beta * self.c_anw_rec_linear[i](self.linear_drop(mem_c)) + \
                            (1 - beta) * self.z_anw_rec_linear[i](self.linear_drop(mem_z)) + \
                            self.query_rec_linear[i](self.linear_drop(add_z))
                if config.mem_hop > 1:
                    add_z = rec_z

            z_logit = self.bow(rec_z)  # [batch_size, vocab_size] #todo z+meta
            z_logit = z_logit.unsqueeze(1).repeat(1, dec_batch.shape[1], 1)
           
            loss_aux += self.criterion(z_logit.contiguous().view(-1, z_logit.size(-1)), dec_batch.contiguous().view(-1))

            
            emo_logit = self.emo(self.emo_drop(emo_z))
            inf_emo = self.emotion_embedding(emo_logit)
            emo_loss = self.emo_criterion(emo_logit, batch["program_label"] - 9)
            #emo_loss = torch.abs(emo_loss-1.4) + 1.4
            pred_emotion = np.argmax(emo_logit.detach().cpu().numpy(), axis=1)
            emotion_acc = accuracy_score((batch["program_label"] - 9).cpu().numpy(), pred_emotion)
            return_dict['emo_loss'] = emo_loss.item()
            return_dict['emo_acc'] = emotion_acc
        else:
            return_dict['emo_loss'] = 0
            return_dict['emo_acc'] = 0
        if config.rec_weight != 0:
            
            sent_z = torch.cat((rec_z, inf_emo, c_emb), -1)
                
           
            sent_z = self.add_z_linear(self.linear_drop(sent_z))
            input_vector[:, 0] = input_vector[:, 0] + sent_z
            if config.c_r_codebook:
                input_vector[:,0] = input_vector[:,0] + z_c_r

            mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
            mask_src_c_ = batch['atomic_relation'].data.eq(0).squeeze(2).unsqueeze(1)
            mask_src_r_ = batch['atomic_relation_t'].data.eq(0).squeeze(2).unsqueeze(1)

            if config.cross:
                pre_logit, attn_dist = self.decoder(input_vector, (encoder_outputs, atomic_know_rel_hidden, z),
                                                 (mask_src, mask_src_c_, mask_src_r_, mask_trg))
            else:
                pre_logit, attn_dist = self.decoder(input_vector, encoder_outputs, (mask_src, mask_trg))
            ori_pre_logit = pre_logit
            
            logit = self.generator(ori_pre_logit, None, None)
            
            loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
            loss_rec_item = loss_rec.item()
            if config.label_smoothing:
                loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                              dec_batch.contiguous().view(-1)).item()
                loss_rec_item = loss_ppl


       
        if config.rec_weight == 0:
            loss = None
            loss_rec_item = 0
        else:
            loss = config.rec_weight*config.codebook_weight*loss_rec
        if not config.only:
            if loss is None:
                loss = config.dis_weight * config.codebook_weight * dis_loss + config.aux_weight * config.codebook_weight * loss_aux
            else:
                loss += config.dis_weight * config.codebook_weight * dis_loss + config.aux_weight * config.codebook_weight * loss_aux
        if not config.only:
            if config.train_prior_joint:
                kl_weight = min(math.tanh(6 * iter / (config.full_kl_step - 3)) + 1, 1)
                loss += config.codebook_weight * config.kl_ceiling * kl_weight * cls_loss
            if config.multitask and config.emo_lr != 0:
                
                loss += config.emo_lr * emo_loss
        if config.c_r_codebook:
            loss += config.aux_ceiling * loss_c_r_aux
            kl_weight = min(math.tanh(6 * iter / config.full_kl_step - 3) + 1, 1)
            if train:
                loss += config.kl_ceiling * kl_weight * dis_c_r_loss
            return_dict['c_r_aux'] = loss_c_r_aux.item()
            return_dict['c_r_dis_loss'] = dis_c_r_loss.item()
        else:
            return_dict['c_r_aux'] = 0
            return_dict['c_r_dis_loss'] = 0
       
        elbo = loss_rec_item+dis_loss.item()
        return_dict['loss_seq'] = 0
        return_dict['loss_rec'] = loss_rec_item
        return_dict['aux'] = loss_aux.item()
        return_dict['dis_loss'] = dis_loss.item()
        return_dict['elbo'] = elbo
        return_dict['ppl'] = math.exp(min(loss_rec_item, 100))
        return_dict['encoding'] = encoding
        if(train):
            #loss = loss * config.warm_up
            loss.backward()
            # clip gradient
            nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()
            if (config.noam):
                self.optimizer.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
        return return_dict

    def decoder_greedy(self, batch, max_dec_step=50, save=False):
        return_dict = {}
        return_dict = {}
        atomic_know = batch['atomic_know']
        batchsize, atomic_know_num, atomic_know_len = atomic_know.shape
        flat_atomic_know = atomic_know.reshape(batchsize * atomic_know_num, atomic_know_len)
        mask_atomic_know = flat_atomic_know.data.eq(config.PAD_idx).unsqueeze(1)

        atomic_rel_emb = self.rel_emb(batch["atomic_relation"].repeat(1, 1, atomic_know_len)).reshape(
            batchsize * atomic_know_num, atomic_know_len, -1)

        atomic_rel_emb_c = self.rel_t_emb(batch['atomic_relation']).squeeze(2)
        atomic_rel_emb_r = self.rel_t_emb(batch["atomic_relation_t"]).squeeze(2)

        atomic_rel_emb_mask = batch['atomic_relation'].data.eq(-1)
        atomic_know_hidden = self.atomic_know_encoder(self.embedding(flat_atomic_know) + atomic_rel_emb,
                                                      mask_atomic_know)
        atomic_know_rel_hidden = (atomic_know_hidden.reshape(batchsize, atomic_know_num, atomic_know_len, -1))[:, :, 0,
                                 :]
        # atomic_know_rel_hidden_emo = (atomic_know_hidden.reshape(batchsize, atomic_know_num, atomic_know_len, -1))[:, :,
        #                              1,
        #                              :]

        atomic_know_flat = batch['atomic_know_t_set_all']

        atomic_know_t = batch['atomic_know_t']
        atomic_rel_emb_t_mask = batch['atomic_relation_t'].data.eq(0)

        _, atomic_know_t_num, atomic_know_t_len = atomic_know_t.shape
        flat_atomic_know_t = atomic_know_t.reshape(batchsize * atomic_know_t_num, atomic_know_t_len)
        mask_atomic_know_t = flat_atomic_know_t.data.eq(config.PAD_idx).unsqueeze(1)
        atomic_rel_t_emb = self.rel_emb(batch["atomic_relation_t"].repeat(1, 1, atomic_know_t_len)).reshape(
            batchsize * atomic_know_t_num, atomic_know_t_len, -1)
        # todo share seg emb

        atomic_know_t_hidden = self.atomic_know_encoder(self.embedding(flat_atomic_know_t) + atomic_rel_t_emb,
                                                        mask_atomic_know_t)
        atomic_know_t_rel_hidden = atomic_know_t_hidden.reshape(batchsize, atomic_know_t_num, atomic_know_t_len, -1)[:,
                                   :, 0, :]
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        enc_batch_extend_vocab = enc_batch

        posterior_batch = batch["posterior_batch"]
        posterior_batch_emb = self.embedding(posterior_batch)
        mask_res = batch["posterior_batch"].data.eq(config.PAD_idx).unsqueeze(1)
        posterior_mask = self.embedding(batch["posterior_mask"])

        r_encoder_outputs = self.encoder(posterior_batch_emb + posterior_mask, mask_res)
        ## Encode
        mask_src_r = posterior_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        enc_batch_emb = self.embedding(enc_batch)
        encoder_outputs = self.encoder(enc_batch_emb + emb_mask, mask_src)
        # Decode
        c_emb = encoder_outputs[:, 0]
        r_emb = r_encoder_outputs[:, 0]
        # print(batch['atomic_know_t_bow_target_by_rel'].shape)
        _, tgt_len, bow_target_by_rel_len = batch['atomic_know_t_bow_target_by_rel'].shape

        def attn(X, Y, X_mask, Y_mask, attnv, attnw):
            # X bs, src_len
            # Y bs, tgt_len
            src_len = X.shape[1]
            tgt_len = Y.shape[1]
            # print(Y.shape)
            # print(X.shape)
            # print(X_mask.shape)
            tgt_outputs = Y.repeat(1, 1, src_len).view(batchsize, src_len * tgt_len, config.hidden_dim)
            src_outputs = X.repeat(1, tgt_len, 1).view(batchsize, tgt_len * src_len, config.hidden_dim)
            #  bs, 300
            attn_dist = attnv(
                torch.tanh(attnw(torch.cat([tgt_outputs, src_outputs], dim=-1)))).view(batchsize,
                                                                                       tgt_len,
                                                                                       src_len)

            src2tgt = F.softmax(attn_dist + ((X_mask.squeeze(2).unsqueeze(1).repeat(1, tgt_len, 1)) * -10000000000),
                                dim=2)
            # src2tgt = F.softmax(attn_dist)
            return src2tgt

        def attn_r(X, Y, X_mask, Y_mask, attnv, attnw, r_emb, temp=None, hard=True):
            # X bs, src_len
            # Y bs, tgt_len
            src_len = X.shape[1]
            tgt_len = Y.shape[1]
            # print(Y.shape)
            # print(X.shape)
            # print(X_mask.shape)
            tgt_outputs = Y.repeat(1, 1, src_len).view(batchsize, src_len * tgt_len, config.hidden_dim)
            src_outputs = X.repeat(1, tgt_len, 1).view(batchsize, tgt_len * src_len, config.hidden_dim)
            #  bs, 300
            r_emb = r_emb.repeat(1, tgt_len, 1).view(batchsize, tgt_len * src_len, config.hidden_dim)
            b = torch.tanh(attnw(self.linear_drop(torch.cat([tgt_outputs, src_outputs], dim=-1))))  # bs, t*s, dim
            a = attnv(self.linear_drop(r_emb))  # bs, t*s, dim

            if config.dot:
                a = a.reshape(batchsize * tgt_len * src_len, 1, -1)
                b = b.reshape(batchsize * tgt_len * src_len, -1, 1)
                attn_dist = torch.bmm(a, b).view(batchsize,
                                                 tgt_len,
                                                 src_len)
            else:
                a = a.reshape(batchsize * tgt_len * src_len, -1)
                b = b.reshape(batchsize * tgt_len * src_len, -1)
                attn_dist = torch.cosine_similarity(a, b, dim=-1)
                attn_dist = attn_dist.view(batchsize,
                                                 tgt_len,
                                                 src_len)
            if temp is None:
                attn_dist = attn_dist * config.temp
            else:
                attn_dist = attn_dist * temp
            attn_dist = self.attn_dropout(attn_dist)
          
            attn_dist = attn_dist + ((X_mask.squeeze(2).unsqueeze(1).repeat(1, tgt_len, 1)) * -10000000000)
            if hard and config.hard:
                src2tgt = self.gumbel_softmax(attn_dist, 1, True)
            else:
                src2tgt = F.softmax(attn_dist, dim=2)
            # print(src2tgt.shape, src2tgt)
            # print('----')
            # src2tgt = F.softmax(attn_dist)
            return src2tgt

        query_encoder_outputs = self.encoder(self.embedding(batch["sess"]) + self.embedding(batch["sess_mask"]),
                                             batch["sess"].data.eq(config.PAD_idx).unsqueeze(1))
        q_emb = query_encoder_outputs[:, 0]
        
        rel_target_post_emb = atomic_know_t_rel_hidden
        atomic_rel_emb_mask = atomic_rel_emb_mask.type(
            'torch.FloatTensor').cuda()
        attent_src = atomic_know_rel_hidden
        mask_src_2 = atomic_rel_emb_mask.type('torch.FloatTensor').cuda()
        src2tgt = attn_r(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_ck,
                         attnw=self.attnW_ck, r_emb=atomic_rel_emb_c, temp=1, hard=False)
        res_tmp = src2tgt.unsqueeze(3) * (
            attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_ck_post = res_tmp.sum(2)  # bs, tgt_len, dim

        attent_src = encoder_outputs
        mask_src_2 = enc_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_c,
                       attnw=self.attnW_c)
        res_tmp = src2tgt.unsqueeze(3) * (
            attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_c_post = res_tmp.sum(2)  # bs, tgt_len, dim

        attent_src = r_encoder_outputs
        mask_src_2 = posterior_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_r,
                       attnw=self.attnW_r)
        res_tmp = src2tgt.unsqueeze(3) * (
            attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_r_post = res_tmp.sum(2)  # bs, tgt_len, dim

        res_tmp_post = torch.cat((res_tmp_ck_post, res_tmp_c_post, res_tmp_r_post), -1)
        '''

        '''
        rel_target_post_emb = self.rel_t_emb(batch["atomic_relation_t"].reshape(batchsize, self.atomic_rel_num))
        attent_src = atomic_know_rel_hidden
        mask_src_2 = atomic_rel_emb_mask.type('torch.FloatTensor').cuda()
        src2tgt = attn_r(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_ck,
                         attnw=self.attnW_ck, r_emb=atomic_rel_emb_c, temp=1, hard=False)
        res_tmp = src2tgt.unsqueeze(3) * (
            attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_ck = res_tmp.sum(2)  # bs, tgt_len, dim

        attent_src = encoder_outputs
        mask_src_2 = enc_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_c,
                       attnw=self.attnW_c)
        res_tmp = src2tgt.unsqueeze(3) * (
            attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_c = res_tmp.sum(2)  # bs, tgt_len, dim

        attent_src = r_encoder_outputs
        mask_src_2 = posterior_batch.data.eq(config.PAD_idx).unsqueeze(2).type('torch.FloatTensor').cuda()
        src2tgt = attn(attent_src, rel_target_post_emb, X_mask=mask_src_2, Y_mask=None, attnv=self.attnV_r,
                       attnw=self.attnW_r)
        res_tmp = src2tgt.unsqueeze(3) * (
            attent_src.unsqueeze(1).repeat(1, tgt_len, 1, 1))  # bs, tgt_len, src_len, dim
        res_tmp_r = res_tmp.sum(2)  # bs, tgt_len, dim

        res_tmp_pri = torch.cat((res_tmp_ck, res_tmp_c, res_tmp_r), -1)
        res_tmp = res_tmp_post + res_tmp_pri

        if config.cat_r:
            res_tmp = torch.cat((res_tmp, r_emb.unsqueeze(1).repeat(1, tgt_len, 1)), -1)
      
        res_tmp = torch.cat((res_tmp, atomic_know_t_rel_hidden), -1)
        res_tmp = self.codebook_linear(self.linear_drop(res_tmp))
        res_tmp_cls = torch.cat((res_tmp_ck, res_tmp_c), -1)

        if not config.sep_code:
            res_tmp = res_tmp.reshape(batchsize * tgt_len, -1)
        if config.use_prior:
            pred = batch['pri_label']  # bs, rel, 1
            # if config.sep_code:
            #     pred = pred.reshape(batchsize, tgt_len, 1).transpost(0, 1)
        else:
            pred = None

        if config.c_r_codebook:
            if config.use_prior_joint:
                dis_c_r_loss, z_c_r = self.latent(c_emb, None)
            else:
                dis_c_r_loss, z_c_r = self.latent(c_emb, r_emb)
            z_logit = self.bow3(z_c_r)  # [batch_size, vocab_size] #todo z+meta
            z_logit = z_logit.reshape(batchsize, 1, self.vocab_size).repeat(1, dec_batch.shape[1], 1)
            loss_c_r_aux = self.criterion(z_logit.contiguous().view(-1, z_logit.size(-1)),
                                          dec_batch.contiguous().view(-1))

        if config.use_prior_joint:
            clss = []
            for c in range(self.atomic_rel_num):
                clss.append(
                    self.prior_cls[c](res_tmp_cls.reshape(batchsize, tgt_len, -1)[:, c, :]).reshape(batchsize, 1,
                                                                                                    -1))  # 8 * 1 * 400
            cls = torch.cat(clss, 1).view(batchsize * self.atomic_rel_num, -1)

            pred_cls = np.argmax(cls.detach().cpu().numpy(), axis=1)

           
            pred = torch.tensor(pred_cls).cuda().reshape(batchsize, tgt_len, -1)

        if config.sep_code:
            dis_all_loss = None
            zs = None
            encodeings = None
            for i in range(self.atomic_rel_num):
                if config.use_prior or config.use_prior_joint:
                    dis_loss, z, encoding = self.codebook[i](res_tmp[:, i, :], predicted_label=pred[:, i, :])
                else:
                    dis_loss, z, encoding = self.codebook[i](res_tmp[:, i, :], None)
               
                if dis_all_loss is None:
                    dis_all_loss = dis_loss
                    encodeings = encoding.unsqueeze(1)  # bs, 1, 40
                    zs = z.unsqueeze(1)  # bs, 1, dim
                else:
                    zs = torch.cat((zs, z.unsqueeze(1)), 1)
                    encodeings = torch.cat((encodeings, encoding.unsqueeze(1)), 1)
                    dis_all_loss += dis_loss

            encoding = encodeings.reshape(batchsize * tgt_len, -1)
            dis_loss = dis_all_loss
            z = zs.reshape(batchsize, tgt_len, -1)
        else:
            if config.use_prior or config.use_prior_joint:
                dis_loss, z, encoding = self.codebook(res_tmp, predicted_label=pred.reshape(batchsize*tgt_len, -1))
            else:
                dis_loss, z, encoding = self.codebook(res_tmp)
            z = z.reshape(batchsize, tgt_len, -1)

        if save:
            for k_index in range(10):
                res_tmp_logit = self.bow2(z[:1, k_index, :])  # vocab
                bow_res = res_tmp_logit.detach().cpu().numpy()
                b_bow_index = np.argsort(bow_res)[:, -200:]
                p = [bow_res[0][ind] for ind in b_bow_index[0]]
                bow_pred_k = [[self.vocab.index2word[x] for x in bow_index] for bow_index in b_bow_index]
                return_dict['bow_' + str(k_index)] = [bow_pred_k, p]

        batch_vq_label = encoding
        batch_vq_label = batch_vq_label.detach().cpu().numpy().tolist()
        target = torch.tensor([onehot.index(1) for onehot in batch_vq_label]).cuda()

        # print(pred, target)

        if config.use_prior_dis:
            cls_loss = torch.mean((res_tmp_cls - res_tmp) ** 2)
            return_dict['cls_loss'] = cls_loss.item()

        if config.train_prior_joint:

            clss = []

            for c in range(self.atomic_rel_num):
                #     inf_z = c_emb
                #     a2src = attn(atomic_know_rel_hidden, inf_z.unsqueeze(1), atomic_rel_emb_mask, None, self.attnV_cls[0],
                #                  self.attnW_cls[0])
                #     inf_z = inf_z + (a2src.reshape(batchsize, atomic_know_num, 1) * atomic_know_rel_hidden).sum(1)
                #
                #     clss.append(self.prior_cls[c](inf_z).reshape(batchsize, 1,
                #                                                                                         -1))  # 8 * 1 * 400
                clss.append(
                    self.prior_cls[c](res_tmp_cls.reshape(batchsize, tgt_len, -1)[:, c, :]).reshape(batchsize, 1,
                                                                                                    -1))  # 8 * 1 * 400
            cls = torch.cat(clss, 1).view(batchsize * self.atomic_rel_num, -1)
           
            target = target.view(batchsize, self.atomic_rel_num, -1)
            cls_loss = self.cls_criterion(cls, target.view(-1))
            pred_cls = np.argmax(cls.detach().cpu().numpy(), axis=1)
            cls_acc = accuracy_score((target.view(-1)).cpu().numpy(), pred_cls)
            return_dict['c_r_acc'] = 0
            return_dict['loss_vae_cls'] = cls_loss.item()
            return_dict['vae_acc'] = cls_acc
            # return_dict['encoding'] = pred_cls

        
        sos_token = torch.LongTensor([config.SOS_idx] * batchsize).unsqueeze(1).cuda()
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        input_vector = self.embedding(dec_batch_shift)
        if config.multitask:

            add_z = c_emb
            for i in range(config.mem_hop):
                # print('e')
                a2src = attn_r(atomic_know_rel_hidden, add_z.unsqueeze(1), atomic_rel_emb_mask, None,
                               self.attnV_emo_c[i],
                               self.attnW_emo_c[i], atomic_rel_emb_c, temp=1, hard=False)
                if save:
                    print('emo c', a2src)
                mem_c = (a2src.reshape(batchsize, atomic_know_num, 1) * atomic_know_rel_hidden).sum(1)
                a2src = attn_r(z, add_z.unsqueeze(1), atomic_rel_emb_mask, None, self.attnV_emo_z[i],
                               self.attnW_emo_z[i], atomic_rel_emb_r)
                #
                # print(a2src)

                if save:
                    print('emo z', a2src)
                
                mem_z = (a2src.reshape(batchsize, atomic_know_num, 1) * z).sum(1)
                beta = self.gate_emo_linear[i](self.linear_drop(torch.cat((add_z, mem_c, mem_z), -1)))




                   
                if config.softmax:
                    beta = F.softmax(beta, -1)
                    emo_z = beta[:,0:1] * self.c_anw_emo_linear[i](self.linear_drop(mem_c)) + \
                        beta[:,1:2] * self.z_anw_emo_linear[i](self.linear_drop(mem_z)) + \
                        beta[:,2:3] * self.query_emo_linear[i](self.linear_drop(add_z))
                else:
                    beta = torch.sigmoid(beta)
                    emo_z = beta * self.c_anw_emo_linear[i](self.linear_drop(mem_c)) + \
                            (1 - beta) * self.z_anw_emo_linear[i](self.linear_drop(mem_z)) + \
                            self.query_emo_linear[i](self.linear_drop(add_z))
                if config.mem_hop > 1:
                    add_z = emo_z
                if save:
                    print('e', beta)
            add_z = c_emb
            for i in range(config.mem_hop):
                a2src = attn_r(atomic_know_rel_hidden, add_z.unsqueeze(1), atomic_rel_emb_mask, None,
                               self.attnV_rec_c[i],
                               self.attnW_rec_c[i], atomic_rel_emb_c, temp=1, hard=False)
                # print('r')
                if save:
                    print('rec c', a2src)
                # print(a2src)
                mem_c = (a2src.reshape(batchsize, atomic_know_num, 1) * atomic_know_rel_hidden).sum(1)
                a2src = attn_r(z, add_z.unsqueeze(1), atomic_rel_emb_mask, None, self.attnV_rec_z[i],
                               self.attnW_rec_z[i], atomic_rel_emb_r)
                #print(a2src)
                #
               
                mem_z = (a2src.reshape(batchsize, atomic_know_num, 1) * z).sum(1)
                beta = self.gate_rec_linear[i](self.linear_drop(torch.cat((add_z, mem_c, mem_z), -1)))
                #
                if save:
                    print('rec z', a2src)

              
                if config.softmax:
                    beta = F.softmax(beta, -1)
                    rec_z = beta[:,0:1] * self.c_anw_rec_linear[i](self.linear_drop(mem_c)) + \
                            beta[:,1:2] * self.z_anw_rec_linear[i](self.linear_drop(mem_z)) + \
                            beta[:,2:3] * self.query_rec_linear[i](self.linear_drop(add_z))
                else:
                    beta = torch.sigmoid(beta)
                    rec_z = beta * self.c_anw_rec_linear[i](self.linear_drop(mem_c)) + \
                            (1 - beta) * self.z_anw_rec_linear[i](self.linear_drop(mem_z)) + \
                            self.query_rec_linear[i](self.linear_drop(add_z))

                if config.mem_hop > 1:
                    add_z = rec_z
                if save:
                    print('r', beta)
                if save:
                    res_tmp_logit = self.bow(rec_z)  # bs, vocab
                    bow_res = res_tmp_logit.detach().cpu().numpy()
                    b_bow_index = np.argsort(bow_res)[:, -200:]
                    p = [bow_res[0][ind] for ind in b_bow_index[0]]
                    pred_bow = [[self.vocab.index2word[x] for x in bow_index] for bow_index in b_bow_index]
                    return_dict['bow'] = [pred_bow, p]

            emo_logit = self.emo(self.emo_drop(emo_z))

            inf_emo = self.emotion_embedding(emo_logit)


     
       
        sent_z = torch.cat((rec_z, inf_emo, c_emb), -1)
            
       

        sent_z = self.add_z_linear(self.linear_drop(sent_z))

        '''
        '''
        ys = torch.ones(enc_batch.shape[0], 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for j in range(max_dec_step+1):
            input_vector = self.embedding(ys)    # bs, 1, dim
            input_vector[:, 0] = input_vector[:, 0] + sent_z
            if config.c_r_codebook:
                input_vector[:, 0] = input_vector[:, 0] + z_c_r

            mask_src_c_ = batch['atomic_relation'].data.eq(0).squeeze(2).unsqueeze(1)
            mask_src_r_ = batch['atomic_relation_t'].data.eq(0).squeeze(2).unsqueeze(1)
            if config.cross:
                pre_logit, attn_dist = self.decoder(input_vector, (encoder_outputs, atomic_know_rel_hidden, z),
                                                (mask_src, mask_src_c_, mask_src_r_, mask_trg))
            else:
                pre_logit, attn_dist = self.decoder(input_vector, encoder_outputs, (mask_src, mask_trg))
            ori_pre_logit = pre_logit

            
            prob = self.generator(ori_pre_logit, None, None)
            _, next_word = torch.max(prob[:, -1], dim = 1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])

            if config.USE_CUDA:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return_dict['sent'] = sent
        return return_dict
    def forward(self, batch, iter, train=True, mode='train_one_batch'):
        if mode == 'train_one_batch':
            return self.train_one_batch(batch, iter, train)


