import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ParRanker(nn.Module):
    def __init__(self, root, word_list, glove_dict, word_dim, hidden_dim):
        super(ParRanker, self).__init__()
        word_list = np.load(root+word_list)
        glove_dict= torch.from_numpy(np.load(root+glove_dict))

        self.w_emb = nn.Embedding(len(word_list), word_dim, padding_idx=0)
        self.w_emb.weight.data[2:].copy_(glove_dict)
        self.bilstm = nn.LSTM(word_dim, hidden_dim, 3,
                              batch_first=True, bidirectional=True)
        self.sigmoid = nn.Sigmoid()

    def desc_sort(self, x, len):
        len_sort, ind = torch.sort(len, 0, descending=True)
        x_sort = x[ind]
        return x_sort, len_sort, ind

    def forward(self, x, y, len_x, len_y):
        x = self.w_emb(x)
        y = self.w_emb(y)
        x_sort, len_x_sort, ind_x = self.desc_sort(x, len_x)
        y_sort, len_y_sort, ind_y = self.desc_sort(y, len_y)
        x_pack = pack_padded_sequence(x_sort, len_x_sort, batch_first=True)
        y_pack = pack_padded_sequence(y_sort, len_y_sort, batch_first=True)

        x_, hidden_x_ = self.bilstm(x_pack)
        x,_ = pad_packed_sequence(x_, batch_first=True)
        hidden_x,_ = pad_packed_sequence(hidden_x_, batch_first=True)
        print(x.size(), hidden_x.size())
        return torch.mul(u, v)
