import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ParRanker(nn.Module):
    def __init__(self, root, word_list, glove_dict, word_dim, hidden_dim, dropout):
        super(ParRanker, self).__init__()
        self.hidden_dim = hidden_dim

        word_list = np.load(root+word_list)
        glove_dict= torch.from_numpy(np.load(root+glove_dict))

        self.w_emb = nn.Embedding(len(word_list), word_dim, padding_idx=0)
        self.w_emb.weight.data[2:].copy_(glove_dict)
        self.bilstm = nn.LSTM(word_dim, hidden_dim, 3, dropout=dropout,
                              batch_first=True, bidirectional=True)

    def desc_sort(self, x, len):
        len_sort, ind = torch.sort(len, 0, descending=True)
        x_sort = x[ind]
        _, unsort_ind = torch.sort(ind, 0)
        return x_sort, len_sort, ind, unsort_ind

    def forward(self, x, y, len_x, len_y):
        x = self.w_emb(x)
        y = self.w_emb(y)
        x_sort, len_x_sort, ind_x, unsort_ind_x = self.desc_sort(x, len_x)
        y_sort, len_y_sort, ind_y, unsort_ind_y = self.desc_sort(y, len_y)
        x_pack = pack_padded_sequence(x_sort, len_x_sort, batch_first=True)
        y_pack = pack_padded_sequence(y_sort, len_y_sort, batch_first=True)

        x,_ = self.bilstm(x_pack)
        x,_ = pad_packed_sequence(x, batch_first=True)
        x = torch.cat((torch.stack([x[i,len_x_sort[i]-1,:self.hidden_dim] for i in range(x.size(0))]),
                      x[:,0,self.hidden_dim:]), 1)[unsort_ind_x]

        y,_ = self.bilstm(y_pack)
        y,_ = pad_packed_sequence(y, batch_first=True)
        y = torch.cat((torch.stack([y[i,len_y_sort[i]-1,:self.hidden_dim] for i in range(y.size(0))]),
                      y[:,0,self.hidden_dim:]), 1)[unsort_ind_y]

        return torch.sum(torch.mul(x,y), 1)
