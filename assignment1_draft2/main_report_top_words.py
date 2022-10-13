import torch

from utils import get_data, get_n_neighbors

import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    with open("./outputs/results_pickle/word_vector.pkl", "rb") as f:
        word_vector_list = pickle.load(f)

    with open("./outputs/results_pickle/results.pkl", "rb") as f:
        result_index_list = pickle.load(f)

    with open("./outputs/results_pickle/results_small.pkl", "rb") as f:
        result_small_list = pickle.load(f)

    indices_large = [result[1].numpy() for result in result_index_list]
    indices_small = [result[1].numpy() for result in result_small_list]
    data_dict = get_data()

    for i, (idx_large, idx_small) in enumerate(zip(indices_large, indices_small)):
        print("For Method"+str(i+1)+":")
        print("Top 20 changed words:")
        print([data_dict['w'][w_i.item()] for w_i in idx_large])
        print("Top 20 unchanged words:")
        print([data_dict['w'][w_i.item()] for w_i in idx_small])