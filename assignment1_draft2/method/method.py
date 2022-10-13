# Top level class for abstracting the method usages
import torch
import torch.nn.functional as F


class BaseMethod:

    def __init__(self, data_dict):

        self.words = data_dict['w']
        self.time_period = data_dict['d']
        self.embedding = data_dict['E']

    @property
    def method_embedding(self):
        return self.embedding

    def get_word_time_point(self):
        """
        The method calculates cosine similarity between word-vector and word-vector at 0th time
        Then it calculates the mean-shift at ith time period
        :return: result (shape: |Vocab| x |time period|
        """
        result_mat = torch.zeros(len(self.words), len(self.time_period))
        for i in range(len(self.time_period)):
            result_mat[:, i] = 1 - F.cosine_similarity(
                self.method_embedding[:, 0, :], self.method_embedding[:, i, :], dim=1
            )

        # Any word that has null word vector in at least 1 time slot is completely ignored.
        for idx in (self.method_embedding.sum(dim=-1) == 0).nonzero():
            result_mat[idx[0], :] = 0

        result_mean_shift = result_mat.clone()
        for i in range(1, len(self.time_period)):
            result_mean_shift[:, i] = torch.mean(result_mat[:, i:], dim=1) - torch.mean(result_mat[:, :i], dim=1)
        return result_mean_shift, result_mat
