import torch
import torch.nn as nn
import utils.config as config
from transformers import BertForSequenceClassification, BertModel

bert_hidden_size = {'tiny': 128, 'mini': 256, 'small': 512, 'medium': 512, 'base': 768}


# def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
#     """ Build a resized Embedding Module from a provided token Embedding Module.
#         Increasing the size will add newly initialized vectors at the end
#         Reducing the size will remove vectors from the end
#     Args:
#         new_num_tokens: (`optional`) int
#             New number of tokens in the embedding matrix.
#             Increasing the size will add newly initialized vectors at the end
#             Reducing the size will remove vectors from the end
#             If not provided or None: return the provided token Embedding Module.
#     Return: ``torch.nn.Embeddings``
#         Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
#     """
#     if new_num_tokens is None:
#         return old_embeddings
#
#     old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
#     if old_num_tokens == new_num_tokens:
#         return old_embeddings
#
#     # Build new embeddings
#     new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
#     new_embeddings.to(old_embeddings.weight.device)
#
#     # initialize all new embeddings (in particular added tokens)
#     self._init_weights(new_embeddings)
#
#     # Copy word embeddings from the previous weights
#     num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
#     new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
#
#     return new_embeddings


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(config.bert_ckpt.format(config.bert_type_rank))
        self.dropout = nn.Dropout(config.linear_dropout)
        self.classifier = nn.Linear(bert_hidden_size[config.bert_type_rank], config.sep_dim)
    def forward(self, sentences_ids):
        sentences_rep = self.encoder(sentences_ids)[1]
        sentences_logits = self.classifier(self.dropout(sentences_rep))
        return sentences_logits

