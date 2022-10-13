import pickle
import torch

path = r"D:\UofT Courses\Computational Models of Semantic Change\Assignment 1\embeddings\embeddings\\"


def get_data():
    with open(path+"data.pkl", "rb") as f:
        data_dict = pickle.load(f)
    data_dict['E'] = torch.FloatTensor(data_dict['E'])
    return data_dict


def get_shuffled_data():
    data_dict = get_data()
    index = torch.randint_like(data_dict['E'], len(data_dict['d']), dtype=torch.long)
    data_dict['E'] = torch.scatter(data_dict['E'], 1, index, data_dict['E'])
    return data_dict


def get_n_neighbors(word_matrix, n):
    word_matrix = word_matrix.clone()

    for idx in (word_matrix.sum(dim=-1) == 0).nonzero():
        # I don't want words whose embedding is null vector calculated as neighbors
        word_matrix[idx[0], idx[1]] = torch.inf

    word_matrix = torch.swapaxes(word_matrix, 0, 1)
    dist = torch.cdist(word_matrix, word_matrix)
    return torch.swapaxes(torch.topk(dist, n+1, dim=2, largest=False).indices[:, : ,1:], 0, 1)
