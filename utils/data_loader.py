
import torch
import torch.utils.data as data

import logging 
from utils import config
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
import pickle as pkl
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
nltk_stopwords = stopwords.words('english')
from transformers import BertTokenizerFast
stopwords = set(nltk_stopwords).union(spacy_stopwords)

from model.common_layer import write_config

from utils.persona_ed_reader import load_dataset


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab, prior=None, post=None, c_r_post=None, c_r_prior=None):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        tot = 0
        for k, v in self.vocab.word2count.items():
            tot += v
        self.word2w = {}
        for k, v in self.vocab.word2count.items():
            self.word2w[k] = v / tot
        max_f = max(self.word2w.values())
        for k, v in self.vocab.word2count.items():
            self.word2w[k] = -(self.word2w[k] / max_f) + 1
        print(len(self.data['context']))
        trans = {
        "AtLocation": 'at location',
        "CapableOf": 'capable of',
        "Causes": 'causes',
        "CausesDesire": 'causes desire',
        "CreatedBy": 'created by',
        "DefinedAs": 'defined as',
        "DesireOf": 'desire of',
        "Desires": 'desires',
        "HasA": 'has a',
        "HasFirstSubevent": 'has first sub event',
        "HasLastSubevent": 'has last sub event',
        "HasPainCharacter": 'has pain character',
        "HasPainIntensity": 'has pain intensity',
        "HasPrerequisite": 'has pre requisite',
        "HasProperty": 'has property',
        "HasSubEvent": 'has sub event',
        "HasSubevent": 'has subevent',
        "HinderedBy": 'hindered by',
        "InheritsFrom": 'inherits from',
        "InstanceOf": 'instance of',
            "IsA": 'is a',
           "LocatedNear": 'located near',
           "LocationOfAction": 'location of action',
            "MadeOf": 'made of',
            "MadeUpOf": 'made up of',
            "MotivatedByGoal": 'motivated by goal',
            # "NotCapableOf": 'not capable of',
            # "NotDesires": 'not desires',
            # "NotHasA": 'not has a',
            # "NotHasProperty": 'not has property',
            # "NotIsA": 'not is a',
            # "NotMadeOf": "not made of",
            "ObjectUse": 'object use',
            "PartOf": 'part of',
            "ReceivesAction": 'receives action',
            "RelatedTo": 'related to',
            "SymbolOf": 'symbol of',
            "UsedFor": 'used for',
            "isAfter": 'is after',
            "isBefore": 'is before',
            "isFilledBy": 'is filled by',
            'xIntent': 'intent',
             'xNeed': 'need',
             'xReason': 'reason',
             'xAttr': ' attribute',
             'xReact': 'react',
             'xWant': ' want',
             'xEffect': 'effect',
             'oReact': 'other react',
             'oWant': 'other want',
             'oEffect': 'other effect',
             #'': 'default',
        }

      
        if config.small_rel:
            trans = {
                'xReason': 'reason',
                'xEffect': 'effect',
                'xAttr': ' attribute',  #
                'xWant': 'want',
                'oEffect': 'other effect',
                'oWant': 'other want',
                'xNeed': 'need',
                #'Desires': 'desire',
                'xIntent': 'intent',
                'xReact': 'react',  #
                'oReact': 'oReact', #
            }
        self.ori_atomic_relation_map = {
              kv[0]: i+1 for i, kv in enumerate(trans.items())
        }
     
        if config.train_bert:
            self.base_tokenizer = BertTokenizerFast.from_pretrained('../bert_ckpt/bert-base-uncased')
        if c_r_post is not None:
            self.data['c_r_vq_label'] = np.array(c_r_post).reshape(-1, 1)
        if c_r_prior is not None:
            self.data['c_r_pri_label'] = np.array(c_r_prior).reshape(-1, 1)
        if post is not None:
            self.data['vq_label'] = np.array(post).reshape(-1, len(self.ori_atomic_relation_map), 1)
        if prior is not None:
            self.data['pri_label'] = np.array(prior).reshape(-1, len(self.ori_atomic_relation_map), 1)
        if config.dataset=="empathetic":
            self.emo_map = {'surprised': 9, 'excited': 10, 'annoyed': 11, 'proud': 12, 'angry': 13, 'sad': 14, 'grateful': 15, 'lonely': 16, 'impressed': 17, 'afraid': 18, 'disgusted': 19, 'confident': 20, 'terrified': 21, 'hopeful': 22, 'anxious': 23, 'disappointed': 24, 'joyful': 25, 'prepared': 26, 'guilty': 27, 'furious': 28, 'nostalgic': 29, 'jealous': 30, 'anticipating': 31, 'embarrassed': 32, 'content': 33, 'devastated': 34, 'sentimental': 35, 'caring': 36, 'trusting': 37, 'ashamed': 38, 'apprehensive': 39, 'faithful': 40}

        else:
            self.emo_map = {
                'no emotion': 9, 'anger': 10, 'disgust': 11, 'fear': 12, 'happiness': 13, 'sadness': 14, 'surprise': 15
            }
    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        # self.atomic_relation_map = {
        #     k: v for k, v in random.sample(self.ori_atomic_relation_map.items(), 5)
        # }
        self.atomic_relation_map = self.ori_atomic_relation_map
        """Returns one data pair (source and target)."""
        item = {}
        if config.train_prior:
            #item['c_r_vq_label'] = self.data['c_r_vq_label'][index]
            item['vq_label'] = self.data['vq_label'][index]
        if config.use_prior:
            #item['c_r_pri_label'] = self.data['c_r_pri_label'][index]
            item['pri_label'] = self.data['pri_label'][index]
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["atomic"] = self.data['atomic'][index]
        item["atomic_t"] = self.data['atomic_t'][index]
        item['atomic_know_t_target_by_rel'], item[
            'atomic_know_t_bow_target_by_rel'], item['X_know_text'], item['X_know_text_set']= self.preprocess_atomic_know_as_bow_target_by_rel(
            self.data['atomic_know_t'][index], self.data['atomic_relation_t'][index], target=item["target_text"])
        item['atomic_know_t_sep'] = self.preprocess_atomic_know_sep(self.data['atomic_know_t_flat'][index], self.data['atomic_relation_t'][index])
        item['atomic_know'] = self.preprocess_atomic_know(self.data['atomic_know'][index], self.data['atomic_relation'][index])
        item['atomic_know_flat'], item['atomic_know_set'] = self.preprocess_atomic_know_flat(
            self.data['atomic_know_flat'][index], self.data['atomic_relation'][index])
        item['atomic_know_set_all'] = self.preprocess_atomic_know_flat_all(
            self.data['atomic_know_flat'][index], self.data['atomic_relation'][index])
        item['atomic_know_t_flat'], item['atomic_know_t_set'] = self.preprocess_atomic_know_flat(
            self.data['atomic_know_t_flat'][index], self.data['atomic_relation_t'][index])
        item['atomic_know_t_set_all'] = self.preprocess_atomic_know_flat_all(
            self.data['atomic_know_t_flat'][index], self.data['atomic_relation_t'][index])
        item["atomic_relation"] = self.preprocess_atomic_rel(self.data['atomic_relation'][index])
        item['atomic_know_t'] = self.preprocess_atomic_know(self.data['atomic_know_t'][index], self.data['atomic_relation_t'][index])
        item["atomic_relation_t"] = self.preprocess_atomic_rel(self.data['atomic_relation_t'][index])
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)

        item['sess'], item['sess_mask'] = self.preprocess_sess(item["context_text"], item['target_text'])
        item["context"], item["context_mask"] = self.preprocess(item["context_text"], atomic=item['atomic'])
        item["posterior"], item["posterior_mask"] = self.preprocess(arr=[item["target_text"]], posterior=True,
                                                                    atomic=item['atomic_t'], fake=config.train_prior)



        item["target"], item['target_w'] = self.preprocess(item["target_text"], anw=True)
        item['target_bow'] = self.preprocess_tgt_bow(item['target_text'])
        return item

    def preprocess_atomic_know(self, atomic_know, rel):
        X_atomic_know = []
        #print(len(atomic_know))
        #print(atomic_know)
        for i, know in enumerate(atomic_know):
            if rel[i] not in self.atomic_relation_map:
                continue
            if config.ab_a:
                X_know = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                          ['none']]
            else:
                X_know = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                      know]
            #X_know = X_know[:2] + [word for word in X_know[2:] if random.random() < 0.5]
            #print(' '.join([self.vocab.index2word[w] for w in X_know]))
            X_atomic_know.append(X_know)
        return X_atomic_know




    def preprocess_atomic_know_as_bow_target_by_rel(self, atomic_know, rel, target=None):
        X_atomic_know = []
        X_atomic_know_set = []
        X_know_text = []
        X_know_set_text = []
        for i, knows in enumerate(atomic_know):
            if rel[i] not in self.atomic_relation_map:
                continue
          
            if config.stop == 0:
                stops = [w for w in knows if w != 'none' and w in stopwords]
                non_stops = [w for w in knows if w != 'none' and w not in stopwords]
                X_know = list(set(stops)) + non_stops
            elif config.stop == 1:
                stops = [w for w in knows if w != 'none' and w in stopwords]
                non_stops = [w for w in knows if w != 'none' and w not in stopwords]
                X_know = non_stops
            else:
                X_know = [w for w in knows if w != 'none']

            X_know = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                          X_know]
           
            if config.ab_a:
                X_know = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                          ['none']]
            X_know = [w for w in X_know if w not in [42, 43, 45]]
            
            X_atomic_know.append(X_know)
            X_atomic_know_set.append(list(X_know))
            X_know_text.append([self.vocab.index2word[w] for w in X_know if w not in stopwords])
            X_know_set_text.append([self.vocab.index2word[w] for w in list(set(X_know)) if w not in stopwords])
        return X_atomic_know, X_atomic_know_set, X_know_text, X_know_set_text


    def preprocess_atomic_know_flat(self, atomic_know, rel):
        X_atomic_know = []
        X_atomic_know_set = []
        for i, knows in enumerate(atomic_know):
            if rel[i] not in self.atomic_relation_map:
                continue
            X_knows = []
            for know in knows:
                X_know = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                      know]
                X_knows.append(X_know)

            X_atomic_know.append(X_knows)
            X_atomic_know_set.append(list(set([w for s in X_knows for w in s])))
        return X_atomic_know, X_atomic_know_set

    def preprocess_atomic_know_flat_all(self, atomic_know, rel):
        X_atomic_know_set = set()
        for i, knows in enumerate(atomic_know):
            if rel[i] not in self.atomic_relation_map:
                continue
            X_knows = []
            for know in knows:
                X_know = []
                for word in know:
                    if word not in stopwords and word in self.vocab.word2index:
                        X_know += [self.vocab.word2index[word]]
                # X_know = [self.vocab.word2index[word] if word in self.vocab.word2index and word not in stopwords for word in
                #       know]
                X_knows.append(X_know)
            X_atomic_know_set = X_atomic_know_set | set([w for s in X_knows for w in s])
        #print(' '.join([self.vocab.index2word[w] for w in list(X_atomic_know_set)]))
        return torch.LongTensor(list(X_atomic_know_set))

    def preprocess_atomic_know_sep(self, atomic_know, rel):  #bs, rel, num, len, vocab
        res = []
        for i, knows in enumerate(atomic_know):
            if rel[i] not in self.atomic_relation_map:
                continue
            X_knows = []
            for know in knows:
                X_know = []
                for word in know:
                    if word not in stopwords and word in self.vocab.word2index and self.vocab.word2index[word] not in [42,43,45]:
                        X_know += [self.vocab.word2index[word]] + [config.EOS_idx]
                # X_know = [self.vocab.word2index[word] if word in self.vocab.word2index and word not in stopwords for word in
                #       know]
                X_knows.append(X_know)
            res.append(X_knows)
        #print(' '.join([self.vocab.index2word[w] for w in list(X_atomic_know_set)]))
        return res
    def preprocess_tgt_bow(self, arr):
        sequence = [self.vocab.word2index[word] for word in
                        arr if word in self.vocab.word2index]
        return torch.LongTensor(sequence)
    def preprocess(self, arr, anw=False, meta=None, posterior=False, atomic=None, atomic_know=None, fake=False):
        """Converts words to ids."""
        if(anw):
            #print(arr)
            sequence_words = [word if word in self.vocab.word2index else config.UNK_idx for word in arr]
            sequence_words = sequence_words[:199]
            sequence = [self.vocab.word2index[word] for word in sequence_words]
            sequence_w =  [self.word2w[word] for word in sequence_words]
            sequence += [config.EOS_idx]
            sequence_w += [0.5]
            return torch.LongTensor(sequence), torch.FloatTensor(sequence_w)
        else:
            X_dial = [config.CLS1_idx] if posterior else [config.CLS_idx]
            X_mask = [config.CLS1_idx] if posterior else [config.CLS_idx]
            X_dial += [self.vocab.word2index['cls_emo']]
            X_mask += [self.vocab.word2index['cls_emo']]
            if (config.model=="seq2seq" or config.model=="cvae"):
                X_dial = []
                X_mask = []

            for i, sentence in enumerate(arr[-3:]):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                #print(' '.join([self.vocab.index2word[w] for w in X_dial]))
                if not posterior and i == len(arr[-3:]) - 1:
                    spk = self.vocab.word2index["Y"]
                if posterior:
                    spk = self.vocab.word2index["SYS"]

                X_mask += [spk for _ in range(len(sentence))]
            if len(X_dial) > 200:
                X_dial = X_dial[:2] + X_dial[-198:]
                X_mask = X_mask[:2] + X_mask[-198:]

            # print(len(X_mask))
            assert len(X_dial) == len(X_mask)
            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)
    def preprocess_sess(self, arr, arr_p):
        """Converts words to ids."""
        X_dial = [config.CLS1_idx]
        X_mask = [config.CLS1_idx]
        # X_dial += [self.vocab.word2index['cls_emo']]
        # X_mask += [self.vocab.word2index['cls_emo']]
        if (config.model == "seq2seq" or config.model == "cvae"):
            X_dial = []
            X_mask = []
        for i, sentence in enumerate(arr[-1:]):
            X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                       sentence]
            spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
            X_mask += [spk for _ in range(len(sentence))]

        assert len(X_dial) == len(X_mask)
        return torch.LongTensor(X_dial), torch.LongTensor(X_mask)



    def preprocess_emo(self, emotion, emo_map):
        # program = [0]*len(emo_map)
        # program[emo_map[emotion]] = 1
        return 0, emo_map[emotion]
    def preprocess_atomic_rel(self, atomic_relation):
        # program = [0]*len(emo_map)
        # program[emo_map[emotion]] = 1
        return [[self.atomic_relation_map[rel]] for rel in atomic_relation if rel in self.atomic_relation_map]

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long() ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_float(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).float()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_bert(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long() ## padding index 1
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

    def merge_multi(sequences):
        #print(sequences)
        lengths_r = [len(seq) for seq in sequences]
        lengths = [len(seq) for seq_set in sequences for seq in seq_set]
        #print(max(lengths_r))
        #print(max(lengths))
        padded_seqs = torch.ones(len(sequences), max(lengths_r), max(lengths)).long() ## padding index 1
        for i, seq_set in enumerate(sequences):
            for j, seq in enumerate(seq_set):
                end = len(seq)

                padded_seqs[i, j, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths_r, lengths
    def merge_mmulti(sequences):
        #print(sequences)
        lengths_r = [len(seq) for seq in sequences]
        lengths_num = [len(seq) for seq_set in sequences for seq in seq_set]
        lengths = [len(s) for seq_set in sequences for seq in seq_set for s in seq]
        #print(max(lengths_r))
        #print(max(lengths))
        padded_seqs = torch.ones(len(sequences), max(lengths_r), max(lengths_num), max(lengths)).long() ## padding index 1
        for i, seq_set in enumerate(sequences):
            for j, seq in enumerate(seq_set):
                for k, s in enumerate(seq):
                    end = len(s)
                #print(end)
                    padded_seqs[i, j, k, :end] = torch.LongTensor(s[:end])
        return padded_seqs
    def merge_multi_label(sequences):
        #print(sequences)
        lengths_r = [len(seq) for seq in sequences]
        lengths = [len(seq) for seq_set in sequences for seq in seq_set]
        padded_seqs = torch.zeros(len(sequences), max(lengths_r), max(lengths)).long() ## padding index 1
        for i, seq_set in enumerate(sequences):
            for j, seq in enumerate(seq_set):
                end = len(seq)
                padded_seqs[i, j, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths_r, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True) ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    sess, _ = merge(item_info['sess'])
    sess_mask, _ = merge(item_info['sess_mask'])
    atomic_know_t_sep = merge_mmulti(item_info['atomic_know_t_sep'])
    atomic_batch, atomic_know_num, atomic_know_len = merge_multi(item_info['atomic_know'])
    atomic_batch_t, atomic_know_num_t, atomic_know_len_t = merge_multi(item_info['atomic_know_t'])
    #print(item_info['atomic_know_t'])

    atomic_relation_batch, atomic_relation_num_, atomic_relation_len = merge_multi_label(item_info['atomic_relation'])
    atomic_relation_batch_t, _, _ = merge_multi_label(item_info['atomic_relation_t'])
    #print(item_info['atomic_know_t_target_by_rel'])
    atomic_know_t_target_by_rel, _, _ = merge_multi(item_info['atomic_know_t_target_by_rel'])
    atomic_know_t_bow_target_by_rel, _, _ = merge_multi(item_info['atomic_know_t_bow_target_by_rel'])
    input_batch, input_lengths = merge(item_info['context'])
    posterior_batch, posterior_lengths = merge(item_info['posterior'])
    input_mask, input_mask_lengths = merge(item_info['context_mask'])
    posterior_mask, posterior_mask_lengths = merge(item_info['posterior_mask'])
    ## Target
    target_batch, target_lengths = merge(item_info['target'])
    target_w, _ = merge_float(item_info['target_w'])
    target_bow, _ = merge(item_info['target_bow'])
    atomic_know_set, _, _ = merge_multi(item_info['atomic_know_set'])
    atomic_know_t_set, _, _ = merge_multi(item_info['atomic_know_t_set'])
    atomic_know_t_set_all, _ = merge(item_info['atomic_know_t_set_all'])
    atomic_know_set_all, _ = merge(item_info['atomic_know_set_all'])
    if config.train_bert:
        bert_ctx, _ = merge_bert(item_info['bert_ctx'])

    if config.USE_CUDA:
        atomic_know_t_sep = atomic_know_t_sep.cuda()
        atomic_batch_t = atomic_batch_t.cuda()
        atomic_relation_batch_t = atomic_relation_batch_t.cuda()
        atomic_relation_batch = atomic_relation_batch.cuda()
        atomic_batch = atomic_batch.cuda()
        input_batch = input_batch.cuda()
        posterior_batch = posterior_batch.cuda()
        posterior_mask = posterior_mask.cuda()
        input_mask = input_mask.cuda()
        target_batch = target_batch.cuda()
        atomic_know_set = atomic_know_set.cuda()
        atomic_know_t_set = atomic_know_t_set.cuda()
        atomic_know_t_set_all = atomic_know_t_set_all.cuda()
        atomic_know_set_all = atomic_know_set_all.cuda()
        atomic_know_t_target_by_rel = atomic_know_t_target_by_rel.cuda()
        atomic_know_t_bow_target_by_rel = atomic_know_t_bow_target_by_rel.cuda()
        target_bow = target_bow.cuda()
        target_w = target_w.cuda()
        sess = sess.cuda()
        sess_mask = sess_mask.cuda()
    if config.train_bert:
        bert_ctx = bert_ctx.cuda()
    d = {}
    if config.use_prior:
        pri_label, _, _ = merge_multi_label(item_info['pri_label'])
        pri_label = pri_label.cuda()
        #c_r_pri_label = torch.tensor(item_info['c_r_pri_label']).cuda()
        #c_r_pri_label, _ = merge(c_r_pri_label)
        #d['c_r_pri_label'] = c_r_pri_label.cuda()
        d['pri_label'] = pri_label
    if config.train_prior:
        #c_r_vq_label = torch.tensor(item_info['c_r_vq_label']).cuda()
        #c_r_vq_label, _ = merge(c_r_vq_label)
        vq_label, _, _ = merge_multi_label(item_info['vq_label'])
        vq_label = vq_label.cuda()
        #d['c_r_vq_label'] = c_r_vq_label.cuda()
        d['vq_label'] = vq_label
    d['target_w'] = target_w
    d['sess'] = sess
    d['sess_mask'] = sess_mask
    d['target_bow'] = target_bow
    d['atomic_know_t_target_by_rel'] = atomic_know_t_target_by_rel
    d['atomic_know_t_bow_target_by_rel'] = atomic_know_t_bow_target_by_rel
    d['atomic_know_t_sep'] = atomic_know_t_sep
    #print(atomic_know_t_sep.shape)
    d["atomic_know_set_all"] = atomic_know_set_all
    d["atomic_know_t_set_all"] = atomic_know_t_set_all
    d["atomic_know_set"] = atomic_know_set
    d["atomic_know_t_set"] = atomic_know_t_set
    d["atomic_know"] = atomic_batch
    d["atomic_know_t"] = atomic_batch_t
    d["atomic_relation"] = atomic_relation_batch
    d["atomic_relation_t"] = atomic_relation_batch_t
    #print(atomic_batch_t.shape)
    d["input_batch"] = input_batch
    if config.train_bert:
        d['bert_ctx'] = bert_ctx
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["input_mask"] = input_mask
    d["posterior_batch"] = posterior_batch
    d["posterior_lengths"] = torch.LongTensor(posterior_lengths)
    d["posterior_mask"] = posterior_mask
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    ##program
    d["target_program"] = item_info['emotion']
    d["program_label"] = torch.LongTensor(item_info['emotion_label'])
    if config.USE_CUDA:
        d["program_label"] = d["program_label"].cuda()
    ##text
    d['atomic_text'] = item_info['atomic']
    d['atomic_t_text'] = item_info['atomic_t']
    d["input_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["program_txt"] = item_info['emotion_text']
    d['k_target_txt'] = item_info['X_know_text']
    d['k_target_txt_set'] = item_info['X_know_text_set']
    return d 


def prepare_data_seq(batch_size=32):  

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))
    post_tra = None
    post_test = None
    post_dev = None
    pri_tra = None
    pri_test = None
    pri_dev = None
    post_c_r_tra = None
    post_c_r_dev = None
    post_c_r_test = None
    pri_test_c_r = None

    dataset_train = Dataset(pairs_tra, vocab, prior=pri_tra, post=post_tra, c_r_post=post_c_r_tra)
    dataset_valid = Dataset(pairs_val, vocab, prior=pri_dev, post=post_dev, c_r_post=post_c_r_dev)
    dataset_test = Dataset(pairs_tst, vocab, prior=pri_test, post=post_test, c_r_post=post_c_r_test,
                           c_r_prior=pri_test_c_r)

    data_loader_tra_ = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=1,
                                                   shuffle=False, collate_fn=collate_fn)

    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)

    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=False, collate_fn=collate_fn)
    if config.emo_beta:
        sz = batch_size
    else:
        sz = 1
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=sz,
                                                 shuffle=False, collate_fn=collate_fn)
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map), len(dataset_train.ori_atomic_relation_map), data_loader_tra_