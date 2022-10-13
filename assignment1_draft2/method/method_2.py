# In this file we have written methods for identifying word sense change by paper Eger and Mehel
import torch
import torch.nn.functional as F
from .method import BaseMethod
from utils import get_data


def cosine_similarity(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    sim_mt = torch.mm(a_norm, a_norm.transpose(0, 1))
    return sim_mt


class VectorOfSimilarities(BaseMethod):

    def __init__(self, data_dict):
        super().__init__(data_dict)
        self.embedding_sim = torch.zeros(len(self.words), len(self.time_period), len(self.words))
        for i in range(len(self.time_period)):
            self.embedding_sim[:, i, :] = cosine_similarity(self.embedding[:, i, :])

    @property
    def method_embedding(self):
        return self.embedding_sim


if __name__ == "__main__":
    m = VectorOfSimilarities(get_data())
    m.get_word_time_point()