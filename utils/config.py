import os
import logging 
import argparse

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6
SOK_idx = 42
CLS1_idx = 7
Y_idx = 8
if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser()
parser.add_argument('--recover', action='store_true')
parser.add_argument('--mem_hop', type=int, default=3)
parser.add_argument('--cross', action='store_true')
parser.add_argument('--train_prior_joint', action='store_true')
parser.add_argument('--use_prior_joint', action='store_true')
parser.add_argument('--use_prior_dis', action='store_true')
parser.add_argument('--linear_dropout', type=float, default=0)
parser.add_argument('--prior_dropout', action='store_true')
parser.add_argument('--cat_know', action='store_true')

parser.add_argument('--codebook_loss', type=float, default=1)
parser.add_argument('--sep_dim', type=int, default=60)

parser.add_argument('--cond', action='store_true')
parser.add_argument('--test_name', type=str, default='test1')
parser.add_argument('--train_prior', action='store_true')
parser.add_argument('--use_prior', action='store_true')
parser.add_argument("--dataset", type=str, default="mojitalk")
parser.add_argument("--v2", action="store_true")
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=2)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default="save/test/")
parser.add_argument("--save_path_pretrained", type=str, default="save/")
parser.add_argument("--cuda", action="store_true")

parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--basic_learner", action="store_true")
parser.add_argument("--project", action="store_true")
parser.add_argument("--topk", type=int, default=0)
parser.add_argument("--l1", type=float, default=.0)
parser.add_argument("--softmax", action="store_true")
parser.add_argument("--schedule", type=float, default=0)
parser.add_argument('--final_test', action='store_true')
parser.add_argument('--train_bert', action='store_true')
parser.add_argument("--large_decoder", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="seq2seq")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument('--save', action='store_true')
parser.add_argument("--noam", action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--emb_file", type=str)

parser.add_argument('--warmup', type=float, default=1/6)
parser.add_argument('--rec_weight', type=float, default=1)
parser.add_argument('--aux_weight', type=float, default=1)
parser.add_argument('--dis_weight', type=float, default=1)
##cvae
parser.add_argument("--full_kl_step", type=int, default=0)
parser.add_argument("--num_var_layers", type=int, default=0)
parser.add_argument("--kl_ceiling", type=float, default=0.1)
parser.add_argument("--aux_ceiling", type=float, default=1)
parser.add_argument("--load_optim", action="store_true")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
## transformer
parser.add_argument('--ab_a', action='store_true')
parser.add_argument('--stop', type=int, default=2)
parser.add_argument('--only', action='store_true')
parser.add_argument("--hop", type=int, default=4)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--depth", type=int, default=256)
parser.add_argument("--filter", type=int, default=512)
parser.add_argument('--small_rel', action='store_true')
parser.add_argument('--sep_code', action='store_true')
parser.add_argument("--emo_beta", action='store_true')
parser.add_argument("--use_atomic", action="store_true")
parser.add_argument("--sep_atomic", action="store_true")
parser.add_argument('--c_r_codebook', action='store_true')
parser.add_argument('--cat_r', action='store_true')
parser.add_argument('--control', type=float, default=0.25)
parser.add_argument('--bert_ckpt', default='../bert_ckpt/bert-{}-uncased/', help='Bert data (small or base)')
parser.add_argument('--bert_vocab', default='../bert_ckpt/bert-{}-uncased/vocab.txt')
parser.add_argument('--bert_type_rank', default='tiny')
parser.add_argument('--emo_lr', type=float, default=1.0)
parser.add_argument('--codebook_weight', type=float, default=1)
parser.add_argument('--know_drop', type=float, default=0)
parser.add_argument('--emo_iter', type=int, default=8000)
parser.add_argument('--emo_weight', type=float, default=1)
parser.add_argument('--emo_start_lr', type=float, default=0.0001)
parser.add_argument('--factor', type=float, default=0.75)
parser.add_argument('--check_iter', type=int, default=500)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--attn_dropout', type=float, default=0)
parser.add_argument('--temp', type=float, default=1)
parser.add_argument('--ck_temp', type=float, default=1)
parser.add_argument('--print_iter', type=int, default=12000)
parser.add_argument('--dot', action='store_true')
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--hard', action='store_true')
def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


arg = parser.parse_args()
print_opts(arg)
cond = arg.cond
hard = arg.hard
eval_every = arg.eval_every
check_iter = arg.check_iter
emo_weight = arg.emo_weight
ck_temp = arg.ck_temp

attn_dropout = arg.attn_dropout
beta2 = arg.beta2
emo_start_lr = arg.emo_start_lr
warm_up = arg.warmup
print_iter = arg.print_iter
dot = arg.dot

ab_a = arg.ab_a
factor = arg.factor
train_prior_joint = arg.train_prior_joint
use_prior_joint = arg.use_prior_joint

rec_weight = arg.rec_weight
aux_weight = arg.aux_weight
know_drop = arg.know_drop
recover = arg.recover
dis_weight = arg.dis_weight
bert_ckpt = arg.bert_ckpt

bert_vocab = arg.bert_vocab
bert_type_rank = arg.bert_type_rank
sep_dim = arg.sep_dim
train_bert = arg.train_bert
linear_dropout = arg.linear_dropout
prior_dropout = arg.prior_dropout
cat_r = arg.cat_r

only = arg.only


model = arg.model

dataset = arg.dataset
large_decoder = arg.large_decoder
topk = arg.topk
l1 = arg.l1

sep_code = arg.sep_code
small_rel = arg.small_rel
emo_lr = arg.emo_lr
emo_iter = arg.emo_iter
codebook_weight = arg.codebook_weight
basic_learner = arg.basic_learner
multitask = arg.multitask
softmax = arg.softmax
stop = arg.stop
cross = arg.cross
hop = arg.hop
save = arg.save
schedule = arg.schedule
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr
codebook_loss = arg.codebook_loss
beam_size=arg.beam_size
project=arg.project
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm
USE_CUDA = arg.cuda
pointer_gen = arg.pointer_gen
is_coverage = arg.is_coverage
use_oov_emb = arg.use_oov_emb
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = 10000
use_prior_dis = arg.use_prior_dis

emb_file = arg.emb_file or "glove.6B.{}d.txt".format(str(emb_dim))
pretrain_emb = arg.pretrain_emb

save_path = arg.save_path
save_path_pretrained = arg.save_path_pretrained

temp = arg.temp
test = arg.test

emo_beta = arg.emo_beta
control = arg.control
full_kl_step = arg.full_kl_step
### transformer 
mem_hop = arg.mem_hop
heads = arg.heads
depth = arg.depth
filter = arg.filter
test_name = arg.test_name
train_prior = arg.train_prior
use_prior = arg.use_prior
final_test = arg.final_test
v2 = arg.v2
num_var_layers = arg.num_var_layers
kl_ceiling = arg.kl_ceiling
aux_ceiling = arg.aux_ceiling
load_optim = arg.load_optim
gradient_accumulation_steps = arg.gradient_accumulation_steps

label_smoothing = arg.label_smoothing
c_r_codebook = arg.c_r_codebook
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))
collect_stats = False

use_atomic = arg.use_atomic
sep_atomic = arg.sep_atomic

