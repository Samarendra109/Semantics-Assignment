# In this file we have written methods for identifying word sense change
# The idea used here is an improvement over the idea suggested by Eger and Mehel
import torch
import torch.nn.functional as F
from .method import BaseMethod
from utils import get_data


def cosine_similarity(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    sim_mt = torch.mm(a_norm, a_norm.transpose(0, 1))
    return sim_mt


class PCAOnSimilarityVec(BaseMethod):

    def __init__(self, data_dict):
        super().__init__(data_dict)
        self.embedding_v = torch.zeros(len(self.words), len(self.time_period), len(self.words))
        for i in range(len(self.time_period)):
            self.embedding_v[:, i, :] = cosine_similarity(self.embedding[:, i, :])
        self.embedding_v = self.embedding_v.view(-1, len(self.words))
        self.embedding_v -= self.embedding_v.mean(dim=1, keepdims=True)
        _, _, self.embedding_v = torch.pca_lowrank(self.embedding_v, 300)
        self.embedding_v = self.embedding_v.view(len(self.words), len(self.time_period), -1)

    @property
    def method_embedding(self):
        return self.embedding_v


if __name__ == "__main__":
    m = PCAOnSimilarityVec(get_data())
    m.get_word_time_point()