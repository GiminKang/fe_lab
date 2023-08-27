from utils import save_pickle, load_pickle
from sklearn import metrics
from base_svd import SVD
import numpy as np
import torch
import os


def get_model(load_path):
    mf = SVD(None)
    mf.tmp_load_path = load_path
    mf.load_variables()
    return mf


def rec_analysis(total_score, holdings_data, k=20):
    test_data = holdings_data["test_data"]
    indptr = holdings_data["test_indptr"]
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]
    y_true = test_data[2]

    y_score = total_score[test_data[0], test_data[1]]
    # 메모리 문제를 위해 num_partition등분으로 나눠서 계산
    output = {}
    output["mae"] = abs(y_score - y_true).sum() / len(y_true)
    output["rmse"] = np.sqrt(np.square(y_score - y_true).sum() / len(y_true))
    # output["roc_auc"] = sklearn.metrics.roc_auc_score(y_true, y_score)
    # output["avg_precision"] = sklearn.metrics.average_precision_score(y_true, y_score)

    is_first = True
    result_recall = 0
    result_precision = 0
    result_accuracy = 0
    result_map_at_k = 0
    map = 0
    mrr = 0
    for user in range(m):
        tmp_true = y_true[indptr[user]:indptr[user + 1]]
        total_num_relavant = tmp_true.sum()
        tmp_score = y_score[indptr[user]:indptr[user + 1]]
        sorted_index = tmp_score.argsort()[::-1]
        tmp_true = tmp_true[sorted_index]
        tmp_score = tmp_score[sorted_index]
        if is_first:  # MRR계산용
            first_relavant_loc = tmp_true.argmax() + 1
            mrr += 1 / first_relavant_loc / m
            map += metrics.average_precision_score(tmp_true, tmp_score) / m

        TN = (1 - tmp_true[k:]).sum()

        tmp_true = tmp_true[:k]
        # tmp_score = tmp_score[:k]
        # tmp_pred = np.ones(len(tmp_score))
        TP = sum(tmp_true)
        if total_num_relavant == 0:
            recall = 0
        else:
            recall = TP / min(total_num_relavant, k)
        precision = TP / k

        ap_at_k = 0
        for i in (np.where(tmp_true == 1)[0]):
            ap_at_k += tmp_true[:i + 1].sum() / min(total_num_relavant, i + 1) / k

        result_recall += recall / m
        result_precision += precision / m
        result_accuracy += (TP + TN) / n / m
        result_map_at_k += ap_at_k / m
    output["recall@{}".format(k)] = result_recall
    output["precision@{}".format(k)] = result_precision
    output["accuracy@{}".format(k)] = result_accuracy
    output["MAP@{}".format(k)] = result_map_at_k
    if is_first:
        output["MRR"] = mrr
        output["MAP"] = map

    return output


if __name__ == "__main__":
    #model_name = "lgcn"
    model_name = "svd"
    #model_name = "ultra_gcn"

    holdings_data = load_pickle("data.pkl")
    #print(holdings_data)
    model = get_model(f"results/{model_name}_params.pkl")

    m = model.n_users
    n = model.n_items

    total_score = []
    for user in range(m):
        total_score.append(
            model.forward(user, np.arange(model.n_items))
        )
    total_score = np.array(total_score)

    score = torch.load("score.pt")
    score = score.cpu().detach().numpy()

    score2 = torch.load("score2.pt")
    score2 = score2.cpu().detach().numpy()

    print("SVD")
    print(rec_analysis(total_score, holdings_data, k=20))
    print("UltraGCN")
    print(rec_analysis(score, holdings_data, k=20))
    print("HCCFS")
    print(rec_analysis(score2, holdings_data, k=20))
