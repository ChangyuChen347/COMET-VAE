# -*- coding: utf-8 -*-

""" 
emotion2id = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "surprise": 3,
    "fear": 4,
    "anger": 5,
    "disgust": 6
}
"""
from __future__ import print_function, division, unicode_literals
from tqdm import tqdm
import numpy as np
from emotion_classifier.create_torchmoji_embedding_3 import  create_torchmoji_emb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LinearModel(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(feature_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: (batch_size, feature_dim)
        x = self.dropout(x)
        return self.fc(x)

# class LinearModel(nn.Module):
#     def __init__(self, feature_dim, output_dim, dropout):
#         super(LinearModel, self).__init__()
#         self.fc = nn.Linear(feature_dim, 64)
#         self.fc1 = nn.Linear(64, output_dim)
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x):
#         # x: (batch_size, feature_dim)
#         x = self.dropout(x)
#         x = self.fc(x)
#         x = torch.relu(x)
#         return self.fc1(self.dropout(x))

def cls_emo(res_pkl):
    create_torchmoji_emb(res_pkl)
    path = res_pkl + 'ea_res-emb.npy'
    batch_size = 64
    device = torch.device(0)
    dropout=0.3
    model = LinearModel(2304, 6, dropout)
    model.load_state_dict(torch.load("./saved_model/deepmoji.pt"))
    model.to(device)
    load_path = path
    save_path = path.replace("emb.npy", "-emotions.txt")
    valid_embedding = np.load(load_path)
    print("Start inference...")
    valid_iter = DataLoader(valid_embedding, batch_size=batch_size, shuffle=False)
    model.eval()
    output = []
    for batch_x in valid_iter:
        batch_x = torch.FloatTensor(batch_x).to(device)
        with torch.no_grad():
            batch_output = model(batch_x)
        output.extend(batch_output.argmax(dim=1).tolist())
    # save predictions
    print("Saving inference...")
    with open(save_path, "w") as f:
        for emotion in output:
            f.write(str(emotion) + "\n")
    with open('data/gold_emb.npy', 'r') as f:
        x = f.readlines()
    t = 0
    for i in range(len(x)):
        if int(x[i].strip()) == int(output[i]):
            t += 1
    print('EA', t / len(x))