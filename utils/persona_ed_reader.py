import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import time
import nltk
from collections import deque
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
nltk_stopwords = stopwords.words('english')

stopwords = set(nltk_stopwords).union(spacy_stopwords)
class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)  # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



import pickle as pkl
import json

def clean(sentence):
    word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}

    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
  
    return sentence

def read_langs(vocab):
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


    def merge_dict_2(sufs, split, pre='comet_predict_{}_{}.pkl', pkl_file = None):
        keep_rel =  {
                'xReason': 'reason',
                'xEffect': 'effect',
                'xAttr': ' attribute',
                'xWant': 'want',
                'oEffect': 'other effect',
                'oWant': 'other want',
                'xNeed': 'need',
                'xIntent': 'intent',
                'xReact': 'react',
                'oReact': 'oReact',
            }
        res_list = []
        if pkl_file is None:
            for suf in sufs:
                file_name = pre.format(split, suf)
                dic_list = pkl.load(open(file_name, 'rb'))
                if len(res_list) == 0:
                    res_list = dic_list
                else:
                    for i, dic in enumerate(dic_list):
                        res_list[i] = dict(res_list[i], **dic)
        else:
            res_list = pkl_file
        for i, dic in enumerate(tqdm(res_list)):
            dic = {k:v for k,v in dic.items() if k in keep_rel}
            res_list[i] = dic
            # n_dic = {tail.strip(): k for k,v in dic.items() for tail in v}
            # dic = {k: [] for k, v in dic.items()}
            # for v, k in n_dic.items():
            #     dic[k].append(v)
            for k, v in dic.items():
                tail_set = set()
                for tail in v:
                    tail = tail.strip()
                    if tail == 'none' or tail == 'None' or len(tail) == 0:
                        continue
                    tail = tail.replace('.', '')
                    tail = tail.replace('!', '')
                    tail = tail.replace('?', '')
                    tail_set.add(tail)
                tail_set = list(tail_set)
                tail_list = []
                for tail in tail_set:
                    tail = clean(tail)
                    vocab.index_words(tail)
                    tail_list.append(tail)
                res_list[i][k] = tail_list
        return res_list

    def merge_dict_3(file_name):
        keep_rel = {
            'xReason': 'reason',
            'xEffect': 'effect',
            'xAttr': ' attribute',
            'xWant': 'want',
            'oEffect': 'other effect',
            'oWant': 'other want',
            'xNeed': 'need',
            # 'Desires': 'desire',
            'xIntent': 'intent',
            'xReact': 'react',
            'oReact': 'oReact',
        }

        dic_list = pkl.load(open(file_name, 'rb'))
        res_list = dic_list

        for i, dic in enumerate(tqdm(res_list)):
            dic = {k: v for k, v in dic.items() if k in keep_rel}
            res_list[i] = dic
            # n_dic = {tail.strip(): k for k,v in dic.items() for tail in v}
            # dic = {k: [] for k, v in dic.items()}
            # for v, k in n_dic.items():
            #     dic[k].append(v)
            for k, v in dic.items():
                tail_set = set()
                for tail in v:
                    tail = tail.strip()
                    if tail == 'none' or tail == 'None' or len(tail) == 0:
                        continue
                    tail = tail.replace('.', '')
                    tail = tail.replace('!', '')
                    tail = tail.replace('?', '')
                    tail_set.add(tail)
                tail_set = list(tail_set)
                tail_list = []
                for tail in tail_set:
                    tail = clean(tail)
                    vocab.index_words(tail)
                    tail_list.append(tail)
                res_list[i][k] = tail_list
        return res_list



    if config.dataset == 'daily':
        prefix = 'data/daily/'
        train_context = pkl.load(open(prefix + 'train_context.pkl', 'rb'))
        dev_context = pkl.load(open(prefix + 'dev_context.pkl', 'rb'))
        test_context = pkl.load(open(prefix + 'test_context.pkl', 'rb'))
        train_target = pkl.load(open(prefix + 'train_target.pkl', 'rb'))
        test_target = pkl.load(open(prefix + 'test_target.pkl', 'rb'))
        dev_target = pkl.load(open(prefix + 'dev_target.pkl', 'rb'))

        train_emotion = pkl.load(open(prefix + 'train_emotion.pkl', 'rb'))
        test_emotion = pkl.load(open(prefix + 'test_emotion.pkl', 'rb'))
        dev_emotion = pkl.load(open(prefix + 'dev_emotion.pkl', 'rb'))

        # train_situation = ["" for _ in range(len(train_target))]
        # test_situation = ["" for _ in range(len(test_target))]
        # dev_situation = ["" for _ in range(len(dev_target))]

        train_text = open(prefix + 'train.txt').readlines()
        valid_text = open(prefix + 'valid.txt').readlines()
        test_text = open(prefix + 'test.txt').readlines()
        train_text = [t.strip().split('__eou__') for t in train_text]
        valid_text = [t.strip().split('__eou__') for t in valid_text]
        test_text = [t.strip().split('__eou__') for t in test_text]
        train_sess_len = [len(t)-1 for t in train_text]
        valid_sess_len = [len(t)-1 for t in valid_text]
        test_sess_len = [len(t)-1 for t in test_text]
        all_train_atomic = pkl.load(open(prefix+'train_0.comet.pkl', 'rb')) + pkl.load(open(prefix+'train_1.comet.pkl', 'rb')) + pkl.load(open(prefix+'train_2.comet.pkl', 'rb'))
        all_valid_atomic = pkl.load(open(prefix+'valid_0.comet.pkl', 'rb'))
        all_test_atomic = pkl.load(open(prefix+'test_0.comet.pkl', 'rb'))
   
        def split_atomic(sess_len, context, all_atomic):
            atomic = []
            atomic_t = []
            j = 0
            sess_idx = 0
            offset = 0
            for _ in context:
                if j == sess_len[sess_idx] - 1:
                    j = 0
                    offset += sess_len[sess_idx]
                    sess_idx += 1
                atomic.append(all_atomic[offset+j])
                atomic_t.append(all_atomic[offset+j+1])
                j += 1
            if j == sess_len[sess_idx] - 1:
                j = 0
                offset += sess_len[sess_idx]
                sess_idx += 1

            assert sess_idx == len(sess_len)
            return atomic, atomic_t
        train_atomic, train_atomic_t = split_atomic(train_sess_len, train_context, all_train_atomic)
        dev_atomic, dev_atomic_t = split_atomic(valid_sess_len, dev_context, all_valid_atomic)
        test_atomic, test_atomic_t = split_atomic(test_sess_len, test_context, all_test_atomic)

        train_atomic = merge_dict_2([], 'train', pkl_file=train_atomic)
        train_atomic_t = merge_dict_2([], 'train', pkl_file=train_atomic_t)
        dev_atomic = merge_dict_2([], 'dev', pkl_file=dev_atomic)
        dev_atomic_t = merge_dict_2([], 'dev', pkl_file=dev_atomic_t)
        test_atomic = merge_dict_2([], 'test', pkl_file=test_atomic)
        test_atomic_t = merge_dict_2([], 'test', pkl_file=test_atomic_t)



        train_idx = pkl.load(open(prefix + 'train_idx.pkl', 'rb'))
        test_idx = pkl.load(open(prefix + 'test_idx.pkl', 'rb'))
        dev_idx = pkl.load(open(prefix + 'val_idx.pkl', 'rb'))
        print(len(train_context), len(dev_context), len(test_context))

        train_context = [train_context[i] for i in train_idx]
        test_context = [test_context[i] for i in test_idx]
        dev_context = [dev_context[i] for i in dev_idx]
        train_target = [train_target[i] for i in train_idx]
        test_target = [test_target[i] for i in test_idx]
        dev_target = [dev_target[i] for i in dev_idx]
        print(len(train_context), len(dev_context), len(test_context))

        train_emotion = [train_emotion[i] for i in train_idx]
        test_emotion = [test_emotion[i] for i in test_idx]
        dev_emotion = [dev_emotion[i] for i in dev_idx]
        train_situation = ["" for _ in range(len(train_target))]
        test_situation = ["" for _ in range(len(test_target))]
        dev_situation = ["" for _ in range(len(dev_target))]
        print(len(train_atomic), len(dev_atomic), len(test_atomic))
        train_atomic = [train_atomic[i] for i in train_idx]
        train_atomic_t = [train_atomic_t[i] for i in train_idx]
        dev_atomic = [dev_atomic[i] for i in dev_idx]
        dev_atomic_t = [dev_atomic_t[i] for i in dev_idx]
        test_atomic = [test_atomic[i] for i in test_idx]
        test_atomic_t = [test_atomic_t[i] for i in test_idx]
        print(len(train_atomic), len(dev_atomic), len(test_atomic))
    else:
        train_context = np.load('data/empathetic-dialogue/sys_dialog_texts.train.npy')
        train_target = np.load('data/empathetic-dialogue/sys_target_texts.train.npy')
        train_emotion = np.load('data/empathetic-dialogue/sys_emotion_texts.train.npy')
        train_situation = np.load('data/empathetic-dialogue/sys_situation_texts.train.npy')
        #stat(['ccres', 'rrres'], 'train')
        train_atomic = merge_dict_2(['ccres'], 'train')
        train_atomic_t = merge_dict_2(['rrres'], 'train')
        dev_context = np.load('data/empathetic-dialogue/sys_dialog_texts.dev.npy')
        dev_target = np.load('data/empathetic-dialogue/sys_target_texts.dev.npy')
        dev_emotion = np.load('data/empathetic-dialogue/sys_emotion_texts.dev.npy')
        dev_situation = np.load('data/empathetic-dialogue/sys_situation_texts.dev.npy')
        dev_atomic = merge_dict_2(['ccres'], 'dev')
        dev_atomic_t = merge_dict_2(['rrres'], 'dev')
        test_context = np.load('data/empathetic-dialogue/sys_dialog_texts.test.npy')
        test_target = np.load('data/empathetic-dialogue/sys_target_texts.test.npy')
        test_emotion = np.load('data/empathetic-dialogue/sys_emotion_texts.test.npy')
        test_situation = np.load('data/empathetic-dialogue/sys_situation_texts.test.npy')
        test_atomic = merge_dict_2(['ccres'], 'test')
        test_atomic_t = merge_dict_2(['rrres'], 'test')


    print('finish loading')


    data_train = {'context':[],'target':[],'emotion':[], 'situation':[], 'atomic_t': [], 'atomic': [], 'atomic_relation': [], 'atomic_know': [], 'atomic_relation_t': [], 'atomic_know_t': [], 'atomic_know_flat': [], 'atomic_know_t_flat': []}
    data_dev = {'context':[],'target':[],'emotion':[], 'situation':[], 'atomic_t': [], 'atomic': [], 'atomic_relation': [], 'atomic_know': [], 'atomic_relation_t': [], 'atomic_know_t': [], 'atomic_know_flat': [], 'atomic_know_t_flat': []}
    data_test = {'context':[],'target':[],'emotion':[], 'situation':[], 'atomic_t': [], 'atomic': [], 'atomic_relation': [], 'atomic_know': [], 'atomic_relation_t': [], 'atomic_know_t': [], 'atomic_know_flat': [], 'atomic_know_t_flat': []}
    print(len(vocab.index2word))
    max_len = 3
    for context in tqdm(train_context):
        u_list = deque([], maxlen=max_len)
        for u in context:
            u = clean(u)
            u_list.append(u)
            vocab.index_words(u)
        data_train['context'].append(list(u_list))
    for target in train_target:
        target = clean(target)
        data_train['target'].append(target)
        vocab.index_words(target)
    for situation in train_situation:
        situation = clean(situation)
        data_train['situation'].append(situation)
        vocab.index_words(situation)
    for emotion in train_emotion:
        data_train['emotion'].append(emotion)
    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion']) == len(data_train['situation'])
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
    }
    keep_rel = {
        'xReason': 'reason',
        'xEffect': 'effect',
        'xAttr': ' attribute',
        'xWant': 'want',
        'oEffect': 'other effect',
        'oWant': 'other want',
        'xNeed': 'need',
        # 'Desires': 'desire',
        'xIntent': 'intent',
        'xReact': 'react',
        'oReact': 'oReact',
    }
    def filter_conceptnet(conceptnet, vocab):
        filtered_conceptnet = {}
        for k in conceptnet:
            if k in vocab.word2index and k not in stopwords:
                filtered_conceptnet[k] = set()
                for c, w in conceptnet[k]:
                    if c in vocab.word2index and c not in stopwords and w >= 1:
                        filtered_conceptnet[k].add((c, w))
        return filtered_conceptnet

    # remove cases where the same concept has multiple weights
    def remove_KB_duplicates(conceptnet):
        filtered_conceptnet = {}
        for k in conceptnet:
            filtered_conceptnet[k] = set()
            concepts = set()
            filtered_concepts = sorted(conceptnet[k], key=lambda x: x[1], reverse=True)
            for c, w in filtered_concepts:
                if c not in concepts:
                    filtered_conceptnet[k].add((c, w))
                    concepts.add(c)
        return filtered_conceptnet

    def get_filtered_concept():
        dataset = 'EDG'
        conceptnet = pkl.load(open("./data/KB/{0}.pkl".format(dataset), 'rb'))
        #print(len(conceptnet))
        filtered_conceptnet = filter_conceptnet(conceptnet, vocab)
        #print(len(filtered_conceptnet))
        filtered_conceptnet = remove_KB_duplicates(filtered_conceptnet)
      
        return filtered_conceptnet

    #conceptnet = get_filtered_concept()
    print('train_atomic')

    for atomic in tqdm(train_atomic):
        atomic_relation = []
        atomic_know = []
        atomic_know_flat = []
        rel_per_sample = {k: [] for k in keep_rel}
        rel_per_sample = atomic

      
        for k, v in rel_per_sample.items():
            if len(v) == 0:
                continue
            atomic_relation.append(k)
            knows = ['sok_', 'emo_sok']
            for know in v:
                knows += know + ['eok_']
            atomic_know.append(knows)
            atomic_know_flat.append([know for know in v])
        data_train['atomic_relation'].append(atomic_relation)
        data_train['atomic_know'].append(atomic_know)
        data_train['atomic_know_flat'].append(atomic_know_flat)
    print('test_atomic')
    for atomic in test_atomic:
        atomic_relation = []
        atomic_know = []
        atomic_know_flat = []
        rel_per_sample = {k: [] for k in keep_rel}
        rel_per_sample = atomic
     
        for k, v in rel_per_sample.items():
            if len(v) == 0:
                continue
            atomic_relation.append(k)
            knows = ['sok_', 'emo_sok']
            for know in v:
                knows += know + ['eok_']
            atomic_know.append(knows)
            atomic_know_flat.append([know for know in v])
        data_test['atomic_relation'].append(atomic_relation)
        data_test['atomic_know'].append(atomic_know)
        data_test['atomic_know_flat'].append(atomic_know_flat)
    print('dev_atomic')
    for atomic in dev_atomic:
        rel_per_sample = {k: [] for k in keep_rel}
        rel_per_sample = atomic
        atomic_relation = []
        atomic_know_flat = []
        atomic_know = []
   
        for k, v in rel_per_sample.items():
            if len(v) == 0:
                continue
            atomic_relation.append(k)
            knows = ['sok_', 'emo_sok']
            for know in v:
                knows += know + ['eok_']
            atomic_know.append(knows)
            atomic_know_flat.append([know for know in v])
        data_dev['atomic_relation'].append(atomic_relation)
        data_dev['atomic_know'].append(atomic_know)
        data_dev['atomic_know_flat'].append(atomic_know_flat)
    print('train_atomic_t')
    for atomic in train_atomic_t:
        rel_per_sample = {k: [] for k in keep_rel}
        atomic_relation = []
        atomic_know = []
        atomic_know_flat = []
        rel_per_sample = atomic
      
        for k, v in rel_per_sample.items():
            if len(v) == 0:
                continue
            atomic_relation.append(k)
            knows = ['sok_', 'emo_sok']
            for know in v:
                knows += know + ['eok_']
            atomic_know.append(knows)
            atomic_know_flat.append([know for know in v])
        data_train['atomic_relation_t'].append(atomic_relation)
        data_train['atomic_know_t'].append(atomic_know)
        data_train['atomic_know_t_flat'].append(atomic_know_flat)
    print('test_atomic_t')
    for atomic in test_atomic_t:
        rel_per_sample = {k: [] for k in keep_rel}
        atomic_relation = []
        atomic_know = []
        atomic_know_flat = []
        rel_per_sample = atomic
       
        for k, v in rel_per_sample.items():
            if len(v) == 0:
                continue
            atomic_relation.append(k)
            knows = ['sok_', 'emo_sok']
            for know in v:
                knows += know + ['eok_']
            atomic_know.append(knows)
            atomic_know_flat.append([know for know in v])
        data_test['atomic_relation_t'].append(atomic_relation)
        data_test['atomic_know_t'].append(atomic_know)
        data_test['atomic_know_t_flat'].append(atomic_know_flat)
    print('dev_atomic_t')
    for atomic in dev_atomic_t:
        rel_per_sample = {k: [] for k in keep_rel}
        atomic_relation = []
        atomic_know = []
        atomic_know_flat = []
        rel_per_sample = atomic
     
        for k, v in rel_per_sample.items():
            if len(v) == 0:
                continue
            atomic_relation.append(k)
            knows = ['sok_', 'emo_sok']
            for know in v:
                knows += know + ['eok_']
            atomic_know.append(knows)
            atomic_know_flat.append([know for know in v])
        data_dev['atomic_relation_t'].append(atomic_relation)
        data_dev['atomic_know_t'].append(atomic_know)
        data_dev['atomic_know_t_flat'].append(atomic_know_flat)
    print('train_atomic')
    for atomic in train_atomic:
        # u = u + ' SOK'
        u = ['sok_']
        for k, v in atomic.items():
            if k not in keep_rel:
                continue
            for tail in v:
                u += [trans[k]] + tail + ['eok_']
        data_train['atomic'].append(u)


    for atomic in dev_atomic:
        # u = u + ' SOK'
        u = ['sok_']
        for k, v in atomic.items():
            if k not in keep_rel:
                continue
            for tail in v:
                u += [trans[k]] + tail + ['eok_']
        data_dev['atomic'].append(u)

    for atomic in test_atomic:
        # u = u + ' SOK'
        u = ['sok_']
        for k, v in atomic.items():
            if k not in keep_rel:
                continue
            for tail in v:
                u += [trans[k]] + tail + ['eok_']
        data_test['atomic'].append(u)

    for atomic in train_atomic_t:
        # u = u + ' SOK'
        u = ['sok_']
        for k, v in atomic.items():
            if k not in keep_rel:
                continue
            for tail in v:
                u += [trans[k]] + tail + ['eok_']
        data_train['atomic_t'].append(u)
    #stat_r(data_train['context'], data_train['target'], data_train['atomic'], conceptnet)
    print('train_atomic_t')
    for atomic in dev_atomic_t:
        # u = u + ' SOK'
        u = ['sok_']
        for k, v in atomic.items():
            if k not in keep_rel:
                continue
            for tail in v:
                u += [trans[k]] + tail + ['eok_']
        data_dev['atomic_t'].append(u)

    for atomic in test_atomic_t:
        # u = u + ' SOK'
        u = ['sok_']
        for k, v in atomic.items():
            if k not in keep_rel:
                continue
            for tail in v:
                u += [trans[k]] + tail + ['eok_']
        data_test['atomic_t'].append(u)

    for context in dev_context:
        u_list = deque([], maxlen=max_len)
        for u in context:
            u = clean(u)
            u_list.append(u)
            vocab.index_words(u)
        data_dev['context'].append(list(u_list))
    for target in dev_target:
        target = clean(target)
        data_dev['target'].append(target)
        vocab.index_words(target)
    for situation in dev_situation:
        situation = clean(situation)
        data_dev['situation'].append(situation)
        #vocab.index_words(situation)

    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)
    assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion']) == len(data_dev['situation'])

    for context in test_context:
        u_list = deque([], maxlen=max_len)
        for u in context:
            u = clean(u)
            u_list.append(u)
            vocab.index_words(u)
        data_test['context'].append(list(u_list))
    for target in test_target:
        target = clean(target)
        data_test['target'].append(target)
        vocab.index_words(target)
    for situation in test_situation:
        situation = clean(situation)
        data_test['situation'].append(situation)
        #vocab.index_words(situation)
    for emotion in test_emotion:
        data_test['emotion'].append(emotion)
    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['situation'])
    return data_train, data_dev, data_test, vocab

def read_langs_persona(data_train, data_valid, data_test, vocab):
    with open('data/Persona/train_self_original.txt',encoding='utf-8') as f_train:
        context = deque([], maxlen=3)
        for line in f_train:
            line = line.strip()
            nid, line = line.split(' ', 1)
            if '\t' in line:
                u, r, _, cand  = line.split('\t')
                u = clean(u)
                vocab.index_words(u)
                context.append(u)
                data_train['context'].append(list(context))
                r = clean(r)
                vocab.index_words(r)
                data_train['target'].append(r)
                context.append(r)
                data_train['situation'].append(['dummy'])
                data_train['emotion'].append('sentimental')
            else:
                context = deque([], maxlen=3)
    
    with open('data/Persona/valid_self_original.txt',encoding='utf-8') as f_valid:
        context = deque([], maxlen=3)
        for line in f_valid:
            line = line.strip()
            nid, line = line.split(' ', 1)
            if '\t' in line:
                u, r, _, cand  = line.split('\t')
                u = clean(u)
                vocab.index_words(u)
                context.append(u)
                data_valid['context'].append(list(context))
                r = clean(r)
                vocab.index_words(r)
                data_valid['target'].append(r)
                context.append(r)
                data_valid['situation'].append(['dummy'])
                data_valid['emotion'].append('sentimental')
            else:
                context = deque([], maxlen=3)
    
    with open('data/Persona/test_self_original.txt',encoding='utf-8') as f_test:
        context = deque([], maxlen=3)
        for line in f_test:
            line = line.strip()
            nid, line = line.split(' ', 1)
            if '\t' in line:
                u, r, _, cand  = line.split('\t')
                u = clean(u)
                vocab.index_words(u)
                context.append(u)
                data_test['context'].append(list(context))
                r = clean(r)
                vocab.index_words(r)
                data_test['target'].append(r)
                context.append(r)
                data_test['situation'].append(['dummy'])
                data_test['emotion'].append('sentimental')
            else:
                context = deque([], maxlen=3)
    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['situation'])
    return data_train, data_valid, data_test, vocab

def load_dataset():

    if config.dataset == 'empathetic':
        path = 'data/persona_ed/dataset_preproc.p'
    else:
        path = 'data/daily/dataset_preproc.p'
    if(os.path.exists(path)):
        print("LOADING persona_ed")
        with open(path, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab  = read_langs(vocab=Lang({config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS", config.USR_idx:"USR", config.SYS_idx:"SYS", config.CLS_idx:"CLS", config.CLS1_idx:"CLS1", config.Y_idx:"Y",
        9: 'key_surprised', 10: 'key_excited', 11: 'key_annoyed', 12: 'key_proud', 13: 'key_angry', 14: 'key_sad', 15: 'key_grateful', 16: 'key_lonely', 17: 'key_impressed', 18: 'key_afraid', 19: 'key_disgusted', 20: 'key_confident', 21: 'key_terrified', 22: 'key_hopeful',
         23: 'key_anxious', 24: 'key_disappointed', 25: 'key_joyful', 26: 'key_prepared', 27: 'key_guilty', 28: 'key_furious', 29: 'key_nostalgic', 30: 'key_jealous', 31: 'key_anticipating', 32: 'key_embarrassed', 33: 'key_content', 34: 'key_devastated', 35: 'key_sentimental', 36: 'key_caring', 37: 'key_trusting', 38: 'key_ashamed', 39: 'key_apprehensive', 40: 'key_faithful', 41: 'know_', 42: 'sok_', 43: 'eok_', 44: 'cls_emo', 45: 'emo_sok'}))
        #data_tra, data_val, data_tst, vocab = read_langs_persona(data_tra, data_val, data_tst, vocab)
        
        with open(path, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
            
    for i in range(1):
        #print('[situation]:', ' '.join(data_tra['situation'][i]))
        #print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        #print('[atomic]:', ' '.join(data_tra['atomic'][i]))
        print('[target]:', ' '.join(data_tra['target'][i]))
        #print('[atomic_t]:', ' '.join(data_tra['atomic_t'][i]))
        #print('[atomic_know]:', data_tra['atomic_know'][i])
        #print('[atomic_relation]:', data_tra['atomic_relation'][i])
        print(" ")
    # with open("data/persona_ed/dataset_preproc.p", "rb") as f:
    #     _,_,_, vocab = pickle.load(f)
    return data_tra, data_val, data_tst, vocab

