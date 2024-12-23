import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import numpy as np
import pickle as pkl
bert_hidden_size = {'tiny': 128, 'mini': 256, 'small': 512, 'medium': 512, 'base': 768}

class BertEncoder(nn.Module):
    def __init__(self, bert_ckpt, bert_type, adapter=False):
        super(BertEncoder, self).__init__()
       
        self.encoder = BertModel.from_pretrained(bert_ckpt.format(bert_type))
     
        self.dropout = nn.Dropout(0.1)
    def forward(self, sentences_ids, mask=None, seg=None):
        sentences_rep = \
        self.encoder(sentences_ids,
                     attention_mask=mask, token_type_ids=seg)
       
        return sentences_rep[0], sentences_rep[1]


class EmoDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def tgt_tokenize(self, s):
        return self.tokenizer.encode(s, add_special_tokens=False, padding=False, truncation=False)

    def __getitem__(self, index):
        src = self.data[index]['src']
        label = self.data[index]['label']
        src = self.tgt_tokenize(src)
        src = src[:128]
        src = [101] + src + [102]
        return {
            'src': src,
            'label': label,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    def merge_ns(sequences):
        seq_t = [torch.tensor(seq, dtype=torch.int64) for seq in sequences]
        seq_token_ids = nn.utils.rnn.pad_sequence(seq_t, batch_first=True, padding_value=0)
        return seq_token_ids

    def merge_label(labels):
        return torch.tensor(labels, dtype=torch.long)

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    src = merge_ns(item_info['src'])
    label = merge_label(item_info['label'])
    d = {}
    d['src'] = src.cuda()
    d['label'] = label.cuda()
    return d

data_prefix = '/home/v-chanchen/zr_data/'
output_prefix = '/home/v-chanchen/zr_output/'
base_tokenizer = BertTokenizerFast.from_pretrained(data_prefix+'bert_ckpt/bert-base-uncased')

#train_data = np.load('sys_situation_texts.train.npy')

#train_data = np.load('sys_dialog_texts.train.npy', allow_pickle=True)

train_data = np.load('sys_target_texts.train.npy', allow_pickle=True)


# test_data = np.load('sys_situation_texts.test.npy')
#test_data = np.load('sys_dialog_texts.test.npy', allow_pickle=True)
test_data = np.load('sys_target_texts.test.npy', allow_pickle=True)
# train_data = [' '.join(r) for r in train_data]
# test_data = [' '.join(r) for r in test_data]

test_data_gen = pkl.load(open('/home/v-chanchen/VQVAE/all_gen_cmp_pkl', 'rb'))




train_target = np.load('sys_emotion_texts.train.npy')

test_target = np.load('sys_emotion_texts.test.npy')

emo_map = {'surprised': 9, 'excited': 10, 'annoyed': 11, 'proud': 12, 'angry': 13, 'sad': 14, 'grateful': 15,
                'lonely': 16, 'impressed': 17, 'afraid': 18, 'disgusted': 19, 'confident': 20, 'terrified': 21,
                'hopeful': 22, 'anxious': 23, 'disappointed': 24, 'joyful': 25, 'prepared': 26, 'guilty': 27,
                'furious': 28, 'nostalgic': 29, 'jealous': 30, 'anticipating': 31, 'embarrassed': 32, 'content': 33,
                'devastated': 34, 'sentimental': 35, 'caring': 36, 'trusting': 37, 'ashamed': 38, 'apprehensive': 39,
                'faithful': 40}

train_target_id = [emo_map[t]-9 for t in train_target]
test_target_id = [emo_map[t]-9 for t in test_target]
test_data_gen = [[{'src': src.lower(), 'label': test_target_id[i]} for i, src in enumerate(gen_model)] for gen_model in test_data_gen]

train_idx = []
test_idx = []
train_src = {}
test_src = {}
for i, t in enumerate(train_data):
    if t not in train_src:
        train_idx.append(i)
        train_src[t] = 1
for i, t in enumerate(test_data):
    if t not in test_src:
        test_idx.append(i)
        test_src[t] = 1

train_idx = set(train_idx)
test_idx = set(test_idx)
train_data = [{'src': src.lower(), 'label': train_target_id[i]} for i, src in enumerate(train_data) if i in train_idx]

#for t in train_data:


test_data = [{'src': src.lower(), 'label': test_target_id[i]} for i, src in enumerate(test_data) if i in test_idx]


train_data = EmoDataset(train_data, base_tokenizer)
test_data = EmoDataset(test_data, base_tokenizer)
test_data_gen = [EmoDataset(gen_model, base_tokenizer) for gen_model in test_data_gen]
data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=32,
                                               shuffle=True, collate_fn=collate_fn)


data_loader_test = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=1,
                                               shuffle=False, collate_fn=collate_fn)

data_loader_test_gen = [torch.utils.data.DataLoader(dataset=gen_model,
                                               batch_size=32,
                                               shuffle=False, collate_fn=collate_fn) for gen_model in test_data_gen]



def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x




class EmoModel(nn.Module):
    def __init__(self,  data_prefix,  bert_type_rank='base'):
        super().__init__()
        bert_ckpt = data_prefix+'bert_ckpt/bert-{}-uncased/'
        self.encoder = BertEncoder(bert_ckpt=bert_ckpt.format(bert_type_rank), bert_type=bert_type_rank,
                                  )
        self.r_cri = nn.CrossEntropyLoss()
        embed_size = bert_hidden_size[bert_type_rank]
        self.scorer = nn.Linear(embed_size, 32)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        src = batch['src']
        label = batch['label']
        bs = label.shape[0]
        src_mask = ~src.data.eq(0)
        _, enc = self.encoder(src, mask=src_mask)
        logit = self.scorer(self.dropout(enc)).reshape(bs, -1)
        loss = self.r_cri(logit, label.view(-1))
        logits = logit.cpu().detach().numpy()
        predicted_label = np.argmax(logits, axis=1)
        cls_acc = accuracy_score((label.view(-1)).cpu().numpy(), predicted_label)
        return loss, label.view(-1).cpu().numpy(), predicted_label, cls_acc


model = EmoModel(data_prefix).cuda()
optimizer = AdamW(params=model.parameters(), lr=1e-5, eps=1e-8)
warmup_step = 1
last_dev_loss = 100000
max_acc = 0
patient = 0
his_loss = []
his_acc = []
iter = make_infinite(data_loader_train)
training_steps = len(data_loader_train) * 5
steps = tqdm(range(training_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                    num_training_steps=training_steps)
print_every = 100
eval_every = len(data_loader_train) // 2
exp_name = 'emo_dia'

model.load_state_dict(torch.load('/home/v-chanchen/zr_output/emo_2399_emo.tar'), False)
model.eval()
with torch.no_grad():
    predicted_label_list = []
    for gen in data_loader_test_gen:
        predicted_label_ = []
        accs = []
        labels = []
        for dev_batch in gen:
            loss, label, predicted_label, acc = model(dev_batch)
            predicted_label_.extend(predicted_label)
            labels.extend(label)
        predicted_label_list.append(predicted_label_)
        acc_dev = np.mean(accs)
        cls_acc = accuracy_score(labels, predicted_label_)
        cls_acc_gt = accuracy_score(predicted_label_list[0], predicted_label_)
        print('acc_dev', cls_acc)
        print('acc_dev_gt', cls_acc_gt)
