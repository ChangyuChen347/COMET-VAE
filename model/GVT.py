### TAKEN FROM https://github.com/kolloldas/torchnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.common_layer import EncoderLayer, DecoderLayer, VarDecoderLayer, MultiHeadAttention, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask, _get_attn_self_mask,  get_input_from_batch, get_output_from_batch, gaussian_kld,SoftmaxOutputLayer
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


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if(config.pointer_gen):
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist
            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist            
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit, alpha
        else:
            return F.log_softmax(logit,dim=-1)

class Latent(nn.Module):
    def __init__(self,is_eval):
        super(Latent, self).__init__()
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
        self.is_eval = is_eval

    def forward(self,x,x_p, train=True):
        mean = self.mean(x)
        log_var = self.var(x)
        eps = torch.randn(x.size())
        std = torch.exp(0.5 * log_var)
        if config.USE_CUDA: eps = eps.cuda()
        z = eps * std + mean
        kld_loss = 0
        if x_p is not None:
            mean_p = self.mean_p(torch.cat((x_p,x),dim=-1))
            log_var_p = self.var_p(torch.cat((x_p,x),dim=-1))
            kld_loss = gaussian_kld(mean_p,log_var_p,mean,log_var)
            kld_loss = torch.mean(kld_loss)
        if train:
            std = torch.exp(0.5 * log_var_p)
            if config.USE_CUDA: eps = eps.cuda()
            z = eps * std + mean_p
        return kld_loss, z
class CvaeTrans(nn.Module):

    def __init__(self, vocab, emo_number,  model_file_path=None, is_eval=False, load_optim=False, atomic_rel_num=None):
        super(CvaeTrans, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab,config.pretrain_emb)
        self.atomic_know_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)

        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        
        self.r_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        self.decoder = Decoder(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)
        self.attnW = nn.Linear(2 * config.emb_dim, config.emb_dim)
        # self.attnH = nn.Linear(config.emb_dim, config.emb_dim)
        self.attnV = nn.Linear(config.emb_dim, 1, bias=False)
        self.latent_layer = Latent(is_eval)
        self.bow = SoftmaxOutputLayer(config.hidden_dim,self.vocab_size)
        self.atomic_know_cls = SoftmaxOutputLayer(config.hidden_dim, atomic_rel_num+1)
        self.atomic_know_criterion = nn.NLLLoss(ignore_index=0)

        if config.multitask:
            #self.identify = nn.Linear(config.emb_dim, decoder_number, bias=False)
            self.emo = SoftmaxOutputLayer(config.hidden_dim,emo_number)
            self.emotion_embedding = nn.Linear(emo_number, config.hidden_dim)
            self.emo_criterion = nn.NLLLoss()
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
 

        # if model_file_path is not None:
        #     print("loading weights")
        #     state = torch.load(model_file_path, map_location= lambda storage, location: storage)
        #     self.encoder.load_state_dict(state['encoder_state_dict'])
        #     self.r_encoder.load_state_dict(state['r_encoder_state_dict'])
        #     self.decoder.load_state_dict(state['decoder_state_dict'])
        #     self.generator.load_state_dict(state['generator_dict'])
        #     self.embedding.load_state_dict(state['embedding_dict'])
        #     self.latent_layer.load_state_dict(state['latent_dict'])
        #     self.bow.load_state_dict(state['bow'])
        if (config.USE_CUDA):
            self.cuda()
        if is_eval:
            self.eval()
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
            if(config.noam):
                self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])
                if config.USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):

        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'r_encoder_state_dict': self.r_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'latent_dict': self.latent_layer.state_dict(),
            'bow': self.bow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True):
        
        if config.use_atomic and config.sep_atomic:
            atomic_know = batch['atomic_know']
            batchsize, atomic_know_num, atomic_know_len = atomic_know.shape
            flat_atomic_know = atomic_know.reshape(batchsize*atomic_know_num, atomic_know_len)
            mask_atomic_know = flat_atomic_know.data.eq(config.PAD_idx).unsqueeze(1)
            atomic_know_hidden = self.atomic_know_encoder(self.embedding(flat_atomic_know), mask_atomic_know)
            atomic_know_rel_hidden = (atomic_know_hidden.reshape(batchsize, atomic_know_num, atomic_know_len, -1))[:,:,0,:]
            atomic_know_rel_logit = self.atomic_know_cls(atomic_know_rel_hidden)

        
            loss_cls = self.atomic_know_criterion(atomic_know_rel_logit.view(batchsize*atomic_know_num, -1), batch["atomic_relation"].view(-1))
            pred_cls = np.argmax(atomic_know_rel_logit.view(batchsize*atomic_know_num, -1).detach().cpu().numpy(), axis=1)
            cls_acc = accuracy_score((batch["atomic_relation"].view(-1)).cpu().numpy(), pred_cls)

            atomic_know_t = batch['atomic_know_t']
            _, atomic_know_t_num, atomic_know_t_len = atomic_know_t.shape
            flat_atomic_know_t = atomic_know_t.reshape(batchsize * atomic_know_t_num, atomic_know_t_len)
            mask_atomic_know_t = flat_atomic_know_t.data.eq(config.PAD_idx).unsqueeze(1)
            atomic_know_t_hidden = self.atomic_know_encoder(self.embedding(flat_atomic_know_t), mask_atomic_know_t)
            atomic_know_t_rel_hidden = atomic_know_t_hidden.reshape(batchsize, atomic_know_t_num, atomic_know_t_len, -1)[:, :, 0,:]
        else:
            atomic_know_t_num = 0
            atomic_know_num = 0
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        enc_batch_extend_vocab = enc_batch
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        ## Response encode
        posterior_batch = batch["posterior_batch"]
        posterior_batch_emb = self.embedding(posterior_batch)
        mask_res = batch["posterior_batch"].data.eq(config.PAD_idx).unsqueeze(1)
        posterior_mask = self.embedding(batch["posterior_mask"])

        if config.use_atomic and config.sep_atomic:
            posterior_batch_emb = torch.cat([posterior_batch_emb[:,:2,:], atomic_know_t_rel_hidden, posterior_batch_emb[:,2:,:]], dim=1)
            mask_res = torch.cat([torch.ones(batchsize, 1, atomic_know_t_num).eq(0).cuda(), mask_res], dim=-1)
            posterior_mask = torch.cat([posterior_mask[:, :2, :], self.embedding(torch.ones(batchsize, atomic_know_t_num).long().cuda() * 41), posterior_mask[:,2:,:]], dim=1)

        r_encoder_outputs = self.r_encoder(posterior_batch_emb+posterior_mask,mask_res)
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        enc_batch_emb = self.embedding(enc_batch)
        #print(mask_src.shape, emb_mask.shape, enc_batch_emb.shape)
        if config.use_atomic and config.sep_atomic:
            enc_batch_emb = torch.cat([enc_batch_emb[:,:2,:], atomic_know_rel_hidden, enc_batch_emb[:, 2:, :]], dim=1)
            mask_src = torch.cat([torch.ones(batchsize, 1, atomic_know_num).eq(0).cuda(), mask_src], dim=-1)
            emb_mask = torch.cat([emb_mask[:, :2, :], self.embedding(torch.ones(batchsize, atomic_know_num).long().cuda() * 41), emb_mask[:, 2:, :]], dim=1)

        #print(mask_src.shape, emb_mask.shape, enc_batch_emb.shape)
        encoder_outputs = self.encoder(enc_batch_emb+emb_mask,mask_src)
        #latent variable
        if config.model=="cvaetrs":
            kld_loss, z = self.latent_layer(encoder_outputs[:,0], r_encoder_outputs[:,0], train=True)

        # meta = self.embedding(batch["program_label"])
        # if config.dataset=="empathetic":
        #     meta = meta-meta
        # Decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()

        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1) #(batch, len, embedding)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        input_vector = self.embedding(dec_batch_shift)
        emo_context = encoder_outputs[:,1]
        if config.multitask:
            emo_logit = self.emo(emo_context)
            emo_loss = self.emo_criterion(emo_logit, batch["program_label"] - 9)
            pred_emotion = np.argmax(emo_logit.detach().cpu().numpy(), axis=1)
            emotion_acc = accuracy_score((batch["program_label"] - 9).cpu().numpy(), pred_emotion)

        if config.model=="cvaetrs":
            input_vector[:,0] = input_vector[:,0]+z#+meta
        else:
            input_vector[:,0] = input_vector[:,0]#+meta
        if config.multitask:
            input_vector[:,0] = input_vector[:,0]+self.emotion_embedding(emo_logit)
        pre_logit, attn_dist = self.decoder(input_vector,encoder_outputs, (mask_src,mask_trg))


        if not config.sep_atomic and config.pointer_gen:
            tgt_len = pre_logit.shape[1]
            src_len = encoder_outputs.shape[1]
            bsz = encoder_outputs.shape[0]
            attn_pre_logits = pre_logit.repeat(1, 1, src_len).view(bsz, src_len * tgt_len, -1)
            attn_encoder_outputs = encoder_outputs.repeat(1, tgt_len, 1).view(bsz, src_len * tgt_len, -1)
            copy_attn_dist = self.attnV(
                torch.tanh(self.attnW(torch.cat([attn_pre_logits, attn_encoder_outputs], dim=-1)))).view(bsz,
                                                                                                         tgt_len,
                                                                                                         src_len)
        else:
            copy_attn_dist = attn_dist
        ## compute output dist
        logit, alpha = self.generator(pre_logit,copy_attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
        ## loss: NNL if ptr else Cross entropy
        alpha_np = np.mean(alpha.detach().cpu().numpy())
        loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        if config.model=="cvaetrs":
            z_logit = self.bow(z) # [batch_size, vocab_size] #todo z+meta
            z_logit = z_logit.unsqueeze(1).repeat(1,logit.size(1),1)
            loss_aux = self.criterion(z_logit.contiguous().view(-1, z_logit.size(-1)), dec_batch.contiguous().view(-1))

            #kl_weight = min(iter/config.full_kl_step, 0.28) if config.full_kl_step >0 else 1.0
            kl_weight = min(math.tanh(6 * iter/config.full_kl_step - 3) + 1, 1)
            loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss + config.aux_ceiling*loss_aux
            if config.multitask:
                loss = loss_rec + config.kl_ceiling * kl_weight*kld_loss + config.aux_ceiling*loss_aux + emo_loss
            if config.use_atomic and config.sep_atomic:
                loss += loss_cls
            aux = loss_aux.item()
            elbo = loss_rec+kld_loss
        else:
            loss = loss_rec
            elbo = loss_rec
            kld_loss = torch.Tensor([0])
            aux = 0
        if(train):
            loss.backward()
            # clip gradient
            nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()
        if config.multitask:
            return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item(), aux, elbo.item(), emo_loss.item(), emotion_acc, alpha_np
        else:
            return loss_rec.item(), math.exp(
                min(loss_rec.item(), 100)), kld_loss.item(), aux, elbo.item()

    def decoder_greedy(self, batch, max_dec_step=50):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        # meta = self.embedding(batch["program_label"])
        # if config.dataset=="empathetic":
        #     meta = meta-meta
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)
        if config.model=="cvaetrs":
            kld_loss, z = self.latent_layer(encoder_outputs[:,0], None,train=False)
        if config.multitask:
            emo_context = encoder_outputs[:,1]
            emo_logit = self.emo(emo_context)
        ys = torch.ones(enc_batch.shape[0], 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        
        decoded_words = []
        for i in range(max_dec_step+1):
            input_vector = self.embedding(ys)
            # if config.model=="cvaetrs":
            #     input_vector[:,0] = input_vector[:,0]+z#+meta
            # else:
            input_vector[:,0] = input_vector[:,0]#+meta
            if config.multitask:
                input_vector[:, 0] = input_vector[:, 0] + self.emotion_embedding(emo_logit)
            pre_logit, attn_dist= self.decoder(input_vector,encoder_outputs, (mask_src,mask_trg))
            if not config.sep_atomic and config.pointer_gen:
                tgt_len = pre_logit.shape[1]
                src_len = encoder_outputs.shape[1]
                bsz = encoder_outputs.shape[0]
                attn_pre_logits = pre_logit.repeat(1, 1, src_len).view(bsz, src_len * tgt_len, -1)
                attn_encoder_outputs = encoder_outputs.repeat(1, tgt_len, 1).view(bsz, src_len * tgt_len, -1)
                copy_attn_dist = self.attnV(
                    torch.tanh(self.attnW(torch.cat([attn_pre_logits, attn_encoder_outputs], dim=-1)))).view(bsz,
                                                                                                             tgt_len,
                                                                                                             src_len)
            else:
                copy_attn_dist = attn_dist
            prob, _ = self.generator(pre_logit,copy_attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
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
        return sent
