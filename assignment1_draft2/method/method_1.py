# In this file we have written methods for identifying word sense change by paper Kulkarni et al.
from math import sqrt

from .method import BaseMethod
from utils import get_data
import torch
from torch import nn, optim
import torch.nn.functional as F


def get_model(word_vector_size):
    model = nn.Linear(word_vector_size, word_vector_size, bias=False)
    nn.init.eye_(model.weight.data)
    randomness = torch.zeros_like(model.weight.data)
    nn.init.xavier_uniform_(randomness)
    model.weight.data += randomness
    return model


def train_model(model, inputs, labels):
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters())

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        labels = labels.cuda()

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def get_embedding(model, word_vector):
    if torch.cuda.is_available():
        word_vector = word_vector.cuda()
        model = model.cuda()
    return model(word_vector)


def getW(word_time_mat, n, s, t):
    wordmat_s = word_time_mat[:, s, :]
    wordmat_t = word_time_mat[:, t, :]

    model = get_model(word_time_mat.shape[-1])
    train_model(model, wordmat_s, wordmat_t)
    return model


class RotateAlignVectors(BaseMethod):

    def __init__(self, data_dict):
        super().__init__(data_dict)
        self.embedding_rot = torch.zeros_like(self.embedding)
        self.embedding = F.normalize(self.embedding, dim=-1)
        for t in range(len(self.time_period) - 1):
            model = getW(self.embedding, 8, t, len(self.time_period) - 1)
            embedding_t = self.embedding[:, t, :].cuda()
            self.embedding_rot[:, t, :] = model(embedding_t[:, None, :])[:, 0, :].detach()
        self.embedding_rot[:, len(self.time_period) - 1, :] = self.embedding[:, len(self.time_period) - 1, :]

    @property
    def method_embedding(self):
        return self.embedding_rot


if __name__ == "__main__":
    m = RotateAlignVectors(get_data())
    m.get_word_time_point()
