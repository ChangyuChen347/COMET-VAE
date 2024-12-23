# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals

from tqdm import trange
import numpy as np
import json
import pickle as pkl
import sys
sys.path.append("..")
from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_feature_encoding
from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH


def torchmoji_embedding(model, st, sentences, chunksize=5000):
    print(sentences[:3])
    total_embedding = []
    for i in trange(0, len(sentences), chunksize):
        tokenized, _, _ = st.tokenize_sentences(sentences[i:i+chunksize])
        # print(tokenized[:3])
        #print(tokenized)
        total_embedding.extend(model(tokenized))
    return total_embedding


def create_torchmoji_emb(res_pkl):
    file = pkl.load(open(res_pkl, 'rb'))
    maxlen = 30
    print("Loading torchmoji vocab and model...")
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    emoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)
    data = file['hyp_g']
    for i, d in enumerate(data):
        if len(d) == 0:
            data[i] = 'none'
    save_path = res_pkl + 'ea_res-emb.npy'
    embedding = torchmoji_embedding(emoji_model, st, data) # (num_examples, feature_dim)
    print("Saving embedding to {0}".format(save_path))
    np.save(save_path, embedding)
