import torch

from method import RotateAlignVectors, VectorOfSimilarities, PCAOnSimilarityVec
from utils import get_data, get_shuffled_data

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pickle

if __name__ == '__main__':

    torch.manual_seed(1)
    np.random.seed(1)

    method_cls_list = [RotateAlignVectors, VectorOfSimilarities, PCAOnSimilarityVec]
    result_list = []
    result_all_list = []
    result_small_list = []
    result_distribution_list = []
    word_vector_list = []
    result_time_series_list = []

    for method_cls in method_cls_list:

        data_dict = get_data()
        method = method_cls(data_dict)
        result_mean_shift, result_time_series = method.get_word_time_point()
        result_all, time_point = result_mean_shift.max(dim=1)
        result, index = result_all.topk(20)
        result_small, index_small = result_all.topk(20, largest=False)

        word_vector_list.append(method.method_embedding)
        result_all_list.append((result_all, time_point))
        result_list.append((result, index))
        result_small_list.append((result_small, index_small))
        result_time_series_list.append(result_time_series)

        res_distribution = torch.vstack((result, result_small))

        for i in trange(100):
            data_dict = get_shuffled_data()
            method = method_cls(data_dict)
            result_i, _ = method.get_word_time_point()[0].max(dim=1)
            result_i, _ = result_i.topk(20)
            res_distribution = torch.vstack((res_distribution, result_i))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        counts, bins = np.histogram(res_distribution.view(-1).numpy())
        ax.stairs(counts, bins)
        fig.savefig(f'./outputs/method_distribution_plot/{method_cls.__name__}_dist.png')
        plt.close(fig)

        result_distribution_list.append(res_distribution)

    with open("./outputs/results_pickle/result_all.pkl", "wb") as f:
        pickle.dump(result_all_list, f)

    with open("./outputs/results_pickle/results.pkl", "wb") as f:
        pickle.dump(result_list, f)

    with open("./outputs/results_pickle/results_small.pkl", "wb") as f:
        pickle.dump(result_small_list, f)

    with open("./outputs/results_pickle/result_dist.pkl", "wb") as f:
        pickle.dump(result_distribution_list, f)

    with open("./outputs/results_pickle/word_vector.pkl", "wb") as f:
        pickle.dump(word_vector_list, f)

    with open("./outputs/results_pickle/result_time_series.pkl", "wb") as f:
        pickle.dump(result_time_series_list, f)
