import torch

from utils import get_data, get_n_neighbors

import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    with open("./outputs/results_pickle/word_vector.pkl", "rb") as f:
        word_vector_list = pickle.load(f)

    with open("./outputs/results_pickle/results.pkl", "rb") as f:
        result_index_list = pickle.load(f)

    with open("./outputs/results_pickle/result_time_series.pkl", "rb") as f:
        result_time_series_list = pickle.load(f)

    indices = [result[1].numpy() for result in result_index_list][0]
    word_vec = word_vector_list[0]
    result_time_series = result_time_series_list[0]
    data_dict = get_data()

    neighbors = get_n_neighbors(word_vec, 5)
    neighbors_of_top3 = neighbors[indices[:3], :, :]
    result_time_series_top3 = result_time_series[indices[:3], :]

    for idx in indices[:3]:
        word_name = data_dict['w'][idx.item()]
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(
            data_dict['d'],
            result_time_series[idx, :],
            linestyle='--', marker='o'
        )

        for t_i, (t, v) in enumerate(zip(data_dict['d'], result_time_series[idx, :])):
            neighbor_names = [data_dict['w'][n_idx.item()] for n_idx in neighbors[idx, t_i]]
            neighbor_x = t + torch.zeros(5) + 1.5
            neighbor_y = v + torch.sin(torch.tensor([2*i*torch.pi/5 for i in range(5)])) * 0.1
            ax.scatter(neighbor_x, neighbor_y, s=0, c='black')
            for i, name in enumerate(neighbor_names):
                ax.annotate(name, (neighbor_x[i].item(), neighbor_y[i].item()), fontsize=5)

        fig.savefig(f'./outputs/word_semantic_change_plot/{word_name}_dist.png', dpi=600)
        plt.close(fig)
