import numpy as np
import pickle

if __name__ == '__main__':

    with open("./outputs/results_pickle/result_all.pkl", "rb") as f:
        result_all_list = pickle.load(f)

    with open("./outputs/results_pickle/results.pkl", "rb") as f:
        result_index_list = pickle.load(f)

    with open("./outputs/results_pickle/result_dist.pkl", "rb") as f:
        result_distribution_list = pickle.load(f)

    result_all_list = [result_all[0].numpy() for result_all in result_all_list]
    result_list = [result[0].numpy() for result in result_index_list]
    index_list = [result[1].numpy() for result in result_index_list]
    result_distribution_list = [result_distribution.view(-1).numpy()
                                for result_distribution in result_distribution_list]

    for res, dist in zip(result_list, result_distribution_list):

        dist = np.sort(dist)
        res_p_value = dist.searchsorted(res)/dist.shape[0]
        print(res_p_value)

    print(np.corrcoef(np.stack(result_all_list)))