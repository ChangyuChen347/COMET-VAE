from utils.data_loader import prepare_data_seq
from utils import config
from model.seq2seq import SeqToSeq
from model.VQVAE import CvaeTrans
from model.common_layer import evaluate,evaluate_cls, evaluate_tra, count_parameters, make_infinite, get_kld
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import random
seed = 0
torch.manual_seed(seed) # sets the seed for generating random numbers.
torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
opt_level = 'O1'
data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number, atomic_relation_num, data_loader_tra_ = prepare_data_seq(batch_size=config.batch_size)


if(config.test):
    print("Test model",config.model)
    if(config.model == "trs" or config.model == "cvaetrs"):
        model = CvaeTrans(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, is_eval=True)
    # elif(config.model == "cvaetrs"):
    #     model = CvaeTrans(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, is_eval=True)
    elif(config.model == "cvaenad"):
        model = CvaeNAD(vocab,emo_number=program_number, model_file_path=config.save_path_pretrained, is_eval=True)
    elif(config.model == "seq2seq"):
        model = SeqToSeq(vocab, model_file_path=config.save_path_pretrained, is_eval=True)
    elif(config.model == "cvae"):
        model = SeqToSeq(vocab, model_file_path=config.save_path_pretrained, is_eval=True)
    model = model.eval()
    #loss_test, ppl_test, kld_test, bow_test, elbo_test, bleu_score_g, d1,d2,d3 = evaluate(model, data_loader_tst ,ty="test", max_dec_step=50)
    get_kld(model, data_loader_tst ,ty="test", max_dec_step=50)
    exit(0)

if(config.model == "seq2seq"):
    model = SeqToSeq(vocab)
elif(config.model == "cvae"):
    model = SeqToSeq(vocab, model_file_path=config.save_path_pretrained)
elif(config.model == "trs"):
    model = CvaeTrans(vocab,emo_number=program_number)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n !="embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)
elif(config.model == "cvaetrs"):
    model = CvaeTrans(vocab,emo_number=program_number, atomic_rel_num=atomic_relation_num, model_file_path=config.save_path_pretrained, load_optim=config.load_optim)
   # cls_model = CvaeTrans(vocab,emo_number=program_number, atomic_rel_num=atomic_relation_num, model_file_path=config.save_path_pretrained, load_optim=config.load_optim)

elif(config.model == "cvaenad"):
    model = CvaeNAD(vocab,emo_number=program_number)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n !="embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)
print("MODEL USED",config.model)
print("TRAINABLE PARAMETERS",count_parameters(model))
print_every = 250
total_diverse = 0
check_iter = config.check_iter
max_iter = 22000
print_iter = config.print_iter
if config.dataset == 'daily':
    print_every = 500
    check_iter = 500
    max_iter = 30000
    print_iter = 21599
warm = 500


config.warm_up = 1
debug = False
if debug:
    check_iter = 20
total_dis_loss = 0
total_loss_rec = 0
total_c_r_dis_loss = 0
total_c_r_aux_loss = 0
total_loss_seq = 0
total_aux = 0
total_diverse_c_r = 0
total_loss = 0
total_cr_acc = 0
total_acc_vae = 0
total_acc_emo = 0

if config.recover:
    model.load_state_dict(torch.load('{}_vae.tar'.format(config.test_name)), False)
if config.use_prior or config.use_prior_joint:
    with torch.no_grad():
        # evaluate(model, data_loader_tst, ty="valid", max_dec_step=30, do_label=False)
        evaluate(model, data_loader_tst, ty="test", max_dec_step=30, do_label=True)
    exit(0)

last_el = 100000
best_acc_emo = 0
total_emo_loss = 0
last_acc_emo = -10000

exp_w = 1.0
all_res = []
know_drop = config.know_drop
try:
    model.train()
    fit_patient = 0
    best_elbo = 1000
    writer = SummaryWriter(log_dir=config.save_path)
    weights_best = deepcopy(model.state_dict())
    data_iter = make_infinite(data_loader_tra)
    for k in range(1):
        patient = 0
        best_ppl = 100000000000000
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        for n_iter in tqdm(range(1000000)):
            
            config.know_drop = max(0, know_drop - n_iter / 10000 * know_drop)
            model.train()
            if not config.noam:
                if model.optimizer.state_dict()['param_groups'][0][
                'lr'] <= config.emo_start_lr:
                    config.emo_lr = config.emo_weight

            raw = next(data_iter)
            return_dict = model(raw, n_iter, 'train_one_batch')

            batch_vq_label = return_dict['encoding']
            batch_vq_label = batch_vq_label.detach().cpu().numpy().tolist()
            batch_vq_label = [onehot.index(1) for onehot in batch_vq_label]
            total_diverse += len(set(batch_vq_label)) / len(batch_vq_label)
            if config.multitask:
                emo_loss = return_dict['emo_loss']
                acc_emo = return_dict['emo_acc']
                total_acc_emo += acc_emo
                total_emo_loss += emo_loss
            if config.train_prior_joint:
                acc_vae = return_dict['vae_acc']
                cr_vae = return_dict['c_r_acc']
                loss_vae_cls = return_dict['loss_vae_cls']
                total_acc_vae += acc_vae
                total_cr_acc += cr_vae
                total_loss += loss_vae_cls
            if config.use_prior_dis:
                loss_vae_cls = return_dict['cls_loss']
                total_loss += loss_vae_cls
            total_dis_loss += return_dict['dis_loss']
            total_loss_rec += return_dict['loss_rec']
            total_loss_seq += return_dict['loss_seq']
            total_c_r_aux_loss += return_dict['c_r_aux']
            total_c_r_dis_loss += return_dict['c_r_dis_loss']
            total_aux += return_dict['aux']
            if (n_iter+1) % print_every == 0:
                print('C_R_AUX', total_c_r_aux_loss / print_every)
                print('C_R_DIS:', total_c_r_dis_loss / print_every)
                print('DIS:', total_dis_loss / print_every)
                print('SEQ:', total_loss_seq / print_every)
                print('AUX', total_aux / print_every)
                print('REC', total_loss_rec / print_every)
                print('Dist', total_diverse / print_every, len(set(batch_vq_label)), len(batch_vq_label))
                if config.multitask:
                    print('Acc Emo', total_acc_emo / print_every)
                    acc_emo = total_acc_emo / print_every
                    print('Loss Emo', total_emo_loss / print_every)
                    if config.noam:
                        print('lr', model.optimizer.optimizer.state_dict()['param_groups'][0][
                            'lr'])
                    else:
                        print('lr', model.optimizer.state_dict()['param_groups'][0][
                'lr'])
                total_acc_emo = 0
                total_emo_loss = 0
                total_diverse = 0
                total_diverse_c_r = 0
                total_c_r_dis_loss = 0
                total_c_r_aux_loss = 0
                total_dis_loss = 0
                total_loss_rec = 0
                total_aux = 0
                if config.train_prior_joint:
                    print('acc_vae:', total_acc_vae / print_every)
                    print('loss:', total_loss / print_every)
                    print('cr_acc:', total_cr_acc / print_every)
                    total_acc_vae = 0
                    total_loss = 0
                    total_cr_acc = 0
                if config.use_prior_dis:
                    print(total_loss / print_every)
                    total_loss = 0
                print('emo_lr:', config.emo_lr)
                print('warm_up:', config.warm_up)
            t_print_every = 1000
            
            t_print_every = config.eval_every
           
            if((n_iter+1)%check_iter==0):
                model.epoch = n_iter
                model.__id__logger = 0
                with torch.no_grad():
                    loss_val, ppl_val, kld_val, bow_val, elbo_val, bleu_score_g, d1,d2,d3, el, this_res = evaluate(model, data_loader_val ,ty="valid", max_dec_step=30)
                    if not config.noam:
                        model.scheduler.step(ppl_val)
                if config.rec_weight != 0:
                    if ppl_val > best_ppl:
                        patient += 1
                    else:
                        weights_best = deepcopy(model.state_dict())
                        print('save best step {}'.format(n_iter))
                        best_ppl = ppl_val
                        patient = 0
                if n_iter <= 20000:
                    weights_best = deepcopy(model.state_dict())
                    print('save best step {}'.format(n_iter))
                if patient >= 2 and n_iter > max_iter:
                    break
                

                print("[PATIENT]:", patient)
        

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
## TESTING
model.load_state_dict({name: weights_best[name] for name in weights_best})

if config.recover:
    torch.save(weights_best, '{}_vae.tar.recover'.format(config.test_name))
else:
    torch.save(weights_best, '{}_vae.tar'.format(config.test_name))
model.eval()
model.epoch = 100
with torch.no_grad():
    if config.train_prior_joint:
        config.use_prior_joint = True
    evaluate(model, data_loader_tst, ty="test", max_dec_step=30, do_label=True)


