import numpy as np
import random
import argparse
import torch
import json

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder, find_all_metal_names

from utils.data_structure import CompressMethod
from utils.data_visualization import plot_hyperparam_search, plot_compare_cluster, plot_arma_hyper_search, plot_compare_graph
from utils.data_preprocessing import GlobalModifier, load_metal_data, save_date_common
from utils import explore_data, others
from utils.data_structure import DatasetTaskDesc
from experiments import gen_experiment, algo_dict
import matplotlib.pyplot as plt

import warnings
# warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-type", help="Type of Testing (w, f)", type=str)
    parser.add_argument("--iter", help="Number of Training Iteration for GP", type=int)
    args = parser.parse_args()

    test_type = args.test_type
    num_train_iter = args.iter
    create_folder("save")

    length_dataset = 794
    is_test = False
    multi_task_gp_config = algo_dict.encode_params(
        "gp_multi_task", is_verbose=True, 
        is_test=is_test, 
        kernel="Composite_1", 
        optim_iter=num_train_iter,
        len_inp=10
    )

    algo_config = {
        "IIDDataModel": algo_dict.encode_params(
            "iid", is_verbose=False, 
            is_test=is_test, dist="gaussian"
        ), 
        "GPMultiTaskMultiOut": multi_task_gp_config,
        "GPMultiTaskIndex": multi_task_gp_config,
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=False, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=num_train_iter,
            len_inp=10
        ),
        "ARIMAModel": algo_dict.encode_params(
            "arima", is_verbose=False, 
            is_test=is_test, order=(2, 0, 5)
        ),
    } 

    common_compress = CompressMethod(
        # 0, "id", info={"range_index": (0, 260)}
        # 0, "id", info={}
        3, "pca"
    )

    all_modifiers = {
        metal: common_compress
        for metal in find_all_metal_names("data")
    }
    
    task = gen_experiment.gen_task_cluster(
        all_algo=["ARIMAModel"], 
        type_task="time", 
        modifier=all_modifiers, 
        clus_metal_desc="copper",
        clus_time_desc=[[22]],
        algo_config=algo_config,
        len_dataset=130, 
        len_train_show=(274, 130)
    )
    print(task[0][1])

    # [f'v-deep_gp-Matern-{num_train_iter}-10', f'v-deep_gp-Matern-{num_train_iter}-10'], ['arima-8,1,10']
                
    # [f'v-gp_multi_task-Composite_1-{num_train_iter}-10', f'v-gp_multi_task-Composite_1-{num_train_iter}-10'], ['arima-8,1,10']
    
    # [f'v-dspp_gp-Matern-{num_train_iter}-10', f'v-dspp_gp-Matern-{num_train_iter}-10'], ['arima-3,0,3']

    exp_setting2 = {
        "task": {
            "sub_model": [
                [f'v-gp_multi_task-Matern-{num_train_iter}-4', f'v-gp_multi_task-Matern-{num_train_iter}-4'], ['arima-8,1,10']
            ],
            "dataset": [
                [
                    DatasetTaskDesc(
                        inp_metal_list=["copper"],
                        out_feature="copper.Price",
                        out_feat_tran_lag=(22, 0, 'id'),
                        is_drop_nan= False,
                        len_dataset= -1, 
                        metal_modifier=[
                            CompressMethod(compress_dim=3, method='pca', info={})
                        ],
                        use_feature=['Date', 'copper.Feature1', 'copper.Feature2', 'copper.Feature3'],
                        use_feat_tran_lag=[None, None, None, None]
                    ),
                    DatasetTaskDesc(
                        inp_metal_list=["lldpe"],
                        out_feature="lldpe.Price",
                        out_feat_tran_lag=(22, 0, 'id'),
                        is_drop_nan= False,
                        len_dataset= -1, 
                        metal_modifier=[
                            CompressMethod(compress_dim=3, method='pca', info={})
                        ],
                        use_feature=['Date', 'lldpe.Feature1', 'lldpe.Feature2', 'lldpe.Feature3'],
                        use_feat_tran_lag=[None, None, None, None]
                    ),
                ],
                [
                    DatasetTaskDesc(
                        inp_metal_list=["carbon"],
                        out_feature="carbon.Price",
                        out_feat_tran_lag=(22, 0, 'id'),
                        is_drop_nan= False,
                        len_dataset= -1, 
                        metal_modifier=[
                            CompressMethod(compress_dim=0, method='drop', info={})
                        ],
                        use_feature=["Date"],
                        use_feat_tran_lag=[None]
                    ),
                ]
            ],
            "len_pred_show": 130,
            "len_train_show":(200, 100)
        },
        # "algo": ["GPMultiTaskMultiOut", "IndependentMultiModel"],
        # "algo": ["DeepGPMultiOut", "IndependentMultiModel"],
        # "algo": ["DSPPMultiOut", "IndependentMultiModel"],
        # "algo": ["GPMultiTaskIndex", "IndependentMultiModel"],
        # "algo": ["SparseGPIndex", "IndependentMultiModel"],
        "algo": ["DeepGraphGP", "IndependentMultiModel"],
        "using_first": [False, False]
        # "using_first": [True, False]
    }
        
    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting2, is_save=True, is_load=False,
            load_path="Hard_Cluster",
            # load_path="08-25-21-23-05-11-Hard_Cluster"
        )
    elif test_type == "w":
        example_plot_walk_forward(
            exp_setting2, "Hard_Cluster_Walk_Forward",
            is_save=True, is_load=False, is_show=True,
            # load_path="Hard_Cluster_Walk_Forward"
            load_path="08-08-21-09-48-19-Hard_Cluster"
        )
    elif test_type == "r":
        result = gen_experiment.run_experiments(task, save_path="abc")
        print(result)

def fix_hyper_data():
    import os, shutil
    # path = "exp_result/hyper_param_gp/hyper_search_matern"
    # path = "exp_result/hyper_param_gp/hyper_search_matern_periodic"
    # path = "exp_result/hyper_param_gp/hyper_search_rbf"
    path = "exp_result/hyper_param_gp/hyper_search_rbf_periodic"

    get_all_folder = lambda path: sorted([f for f in os.listdir(path) if "." not in f])

    all_path = get_all_folder(path)
    all_run_folder = get_all_folder(path + "/" + all_path[0])

    def merge_json(list_path):
        data = {}
        for path in list_path:
            data.update(others.load_json(path))
        return data

    for run_folder in all_run_folder:
        dest_folder = path + "/" + run_folder
        create_folder(dest_folder)

        all_data_path = []

        for inner_path in all_path:
            from_folder = path + "/" + inner_path + "/" + run_folder
            for folder_content in get_all_folder(from_folder):
                shutil.move(
                    from_folder + "/" + folder_content,
                    dest_folder
                )
            all_data_path.append(from_folder + "/all_data.json")
            
        
        others.dump_json(dest_folder + "/all_data.json", merge_json(all_data_path))
            
    all_final_result_path = []
    for inner_path in all_path:
        search_path = path + "/" + inner_path
        file_path = sorted([f for f in os.listdir(search_path) if ".json" in f])[0]
        search_path += "/" + file_path
        all_final_result_path.append(search_path)
    
    others.dump_json(path + "/" + file_path, merge_json(all_final_result_path))
    
def plot_embedding():
    embedding = np.load("embedding.npy")
    # embedding = embedding[:50]
    num_data = embedding.shape[0]
    embedding = np.reshape(embedding,(-1, embedding.shape[-1]) )
    labels = np.concatenate([
        np.arange(0, 10)
        for i in range(num_data)
    ])
    colors = ["#ff7500", "#5a08bf", "#0062b8", "#1a1a1a", "#20c406", "#ebebeb","#d6022a", "#009688", "#00e5ff", "#1a237e"]


    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import matplotlib

    def plot_3D():
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(embedding)

        fig, ax = plt.subplots(figsize=(10, 10),subplot_kw=dict(projection='3d'))
        x, y, z = reduced_data.T
        ax.scatter3D(
            x, y, z, c=labels, s=10.0, 
            cmap=matplotlib.colors.ListedColormap(colors)
        )
    
    def plot_2D():
        pca = TSNE(n_components=2, perplexity=100)
        reduced_data = pca.fit_transform(embedding).T
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            reduced_data[0], 
            reduced_data[1], 
            c=labels, 
            cmap=matplotlib.colors.ListedColormap(colors),
            zorder=3
        )

        ax.grid(zorder=0)

    plot_2D()
    plt.show()
        
if __name__ == '__main__':
    # plot_embedding()
    # save_date_common("raw_data", "data")

    # fix_hyper_data()

    plot_hyperparam_search()
    # plot_arma_hyper_search("exp_result/hyper_param_arma")
    # plot_compare_cluster()

    # plot_compare_graph()
    
    # example_plot_walk_forward(
    #     {}, "GPMultiTaskMultiOut",
    #     is_save=False, is_load=True, is_show=True,
    #     # load_path="Hard_Cluster_Walk_Forward"
    #     load_path="09-01-21-00-22-08-GPMultiTaskMultiOut"
    # )

    # assert False

    
    # main()
    # gen_experiment.cluster_index_to_nested([0, 3, 0, 2, 3, 1, 1, 1, 2, 4])

