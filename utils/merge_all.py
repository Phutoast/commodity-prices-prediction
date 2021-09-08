import json
import numpy as np
import os

# name_cluster = [
#     "dtw", "dtw-knn", "euclidean", 
#     "euclidean-knn", "expert", 
#     "full_model", "kendell", 
#     "kshape", "peason", "soft-dtw-divergence", 
#     "softdtw-knn", "spearman"
# ]

cluster_compare_path = "no_4_all_wrong_dspp/cluster_compare"
cluster_compare_path = "no_4_all_correct_dspp/cluster_compare"

def get_all_folder(path):
    return [f for f in os.listdir(path) if "." not in f]

def dump_json(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def search_list(l, item):
    for i, data in enumerate(l):
        if data == item:
            return i

    return -1

def cal_bound_results(cluster_compare_path):
    # get all_name cluster !!
    base = load_json(cluster_compare_path + "/compare_cluster.json")
    name_cluster = list(base.keys())
    num_algo = list(base[name_cluster[0]].keys())

    all_data_range = {}

    for cluster in name_cluster:
        true_all_algo = list(base[cluster].keys())
        cluster = "-".join(cluster.split(" "))
        all_data_range[cluster] = {}

        all_sub_path = get_all_folder(cluster_compare_path + f"/{cluster}")
        all_path_algo_name = [p.split("-")[-1] for p in all_sub_path]

        for algo in true_all_algo:
            if not algo in all_path_algo_name:
                all_data_range[cluster][algo] = [None, None]
                # all_data_range[cluster][algo] = None
            else:
                ind = search_list(all_path_algo_name, algo)
                assert ind != -1
                list_all_crps = []
                for task_no in range(10):
                    for fold_num in range(4):
                        test_path = f"{cluster_compare_path}/{cluster}/{all_sub_path[ind]}/task_{task_no}/fold_{fold_num}/loss_detail.json"
                        list_all_crps.append(load_json(test_path)["all_crps"])

                mean = np.mean(list_all_crps)
                bound = np.std(list_all_crps)/np.sqrt(len(list_all_crps))

                if np.isnan(mean) or np.isnan(bound):
                    mean = None
                    bound = None
                all_data_range[cluster][algo] = [mean, bound]
                # all_data_range[cluster][algo] = mean

    dump_json(f"{cluster_compare_path}/bound_compare_cluster.json", all_data_range)

def merge_2_together_replace():
    # Merge 2 together
    all_except_dspp = "no_4_all_wrong_dspp/cluster_compare/bound_compare_cluster.json"
    correct_dspp = "no_4_all_correct_dspp/cluster_compare/bound_compare_cluster.json"

    full_data = {}
    entry = "NonlinearMultiTaskGSPP"
    for k, v in load_json(all_except_dspp).items():
        if k != "full_model":
            print(k)
            correct = load_json(correct_dspp)[k][entry]
            v[entry] = correct

        full_data[k] = v

    dump_json("true_json/compare_cluster_4.json", full_data)

def merge_all_them():
    base_path_1 = "range_gp_no_4/cluster_compare_"
    base_path_2 = "range_deep_gp_no_4/cluster_compare_"

    # for base_path in [base_path_1, base_path_2]:
    #     for i in [2, 3, 5, 6, 7]:
    #         curr_path = base_path + str(i)
    #         cal_bound_results(curr_path)
    #         print(curr_path)

    for i in [2, 3, 5, 6, 7]:
        for j, base_path in enumerate([base_path_1, base_path_2]):
            curr_path = base_path + str(i)
            if j == 0:
                curr_data = load_json(curr_path + "/bound_compare_cluster.json")
            else:
                other_data = load_json(curr_path + "/bound_compare_cluster.json")
                for k, _ in curr_data.items():
                    curr_data[k].update(other_data[k])

        dump_json(f"true_json/compare_cluster_{i}.json", curr_data)


cal_bound_results("fixed_worst_mlt_deep_gp")
# merge_all_them()



