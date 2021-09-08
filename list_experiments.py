import numpy as np
import random
import torch
import os
import json
import argparse
import os.path

from utils.others import create_folder, dump_json, load_json, find_sub_string, find_all_metal_names
from utils.data_visualization import plot_latex, plot_grid_commodity
from utils.data_structure import DatasetTaskDesc, CompressMethod
from utils.data_preprocessing import load_transform_data, parse_series_time, load_metal_data, get_data
from experiments import algo_dict, gen_experiment, metal_desc
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')
 
multi_task_algo = [
    "GPMultiTaskMultiOut", "GPMultiTaskIndex", "IndependentGP"
    "DeepGPMultiOut", "DSPPMultiOut", "SparseGPIndex", 
    "SparseMaternGraphGP", "DeepGraphMultiOutputGP", 
    "DeepGraphInfoMaxMultiOutputGP"
] 
metric = ["MSE", "CRPS"]

default_len_dataset = 794
default_len_train_show = (274, 130)
optim_iter=5

def run_multi_task_gp(save_path, modifier, multi_task_desc, len_inp=10, 
    len_dataset=default_len_dataset, len_train_show=default_len_train_show, 
    kernel="Composite_1", 
    graph_path="exp_result/graph_result/distance correlation_test_graph.npy", 
    is_test=False, is_verbose=False, 
    multi_task_algo=multi_task_algo, is_full_result=False
):
    base_multi_task_fast_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel=kernel, 
        optim_iter=25,
        len_inp=len_inp,
        lr=0.1, 
        graph_path=graph_path,
    )


    base_multi_task_slow_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel=kernel, 
        optim_iter=optim_iter,
        len_inp=len_inp,
        lr=0.05, 
        graph_path=graph_path
    )
    
    base_multi_task_slow_dspp = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel=kernel, 
        optim_iter=3,
        len_inp=len_inp,
        lr=0.1, 
        graph_path=graph_path
    )

    base_multi_task_slower_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel=kernel, 
        optim_iter=500,
        # optim_iter=1,
        len_inp=len_inp,
        lr=0.005, 
        graph_path=graph_path
    )

    default_config = {
        "GPMultiTaskMultiOut": base_multi_task_fast_lr,
        "GPMultiTaskIndex": base_multi_task_fast_lr,
        "DeepGPMultiOut": base_multi_task_slow_lr,
        # Might be wrong.....
        "DSPPMultiOut": base_multi_task_slow_dspp,
        "SparseGPIndex": base_multi_task_slow_lr,
        "SparseMaternGraphGP": base_multi_task_slow_lr,
        "DeepGraphMultiOutputGP": base_multi_task_slower_lr,
        "DeepGraphInfoMaxMultiOutputGP": base_multi_task_fast_lr,
        # Optim = 4 is enough....
        "NonlinearMultiTaskGP": base_multi_task_slow_lr,
        "NonlinearMultiTaskGSPP": base_multi_task_slow_dspp,
        # "DeepGPGraphPropagate": base_multi_task_fast_lr,
        # "DeepGPGraphInteract": base_multi_task_fast_lr,
        # "DSPPGraphInteract": base_multi_task_fast_lr,
        # "DSPPGraphPropagate": base_multi_task_fast_lr,
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=is_verbose, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=optim_iter,
            len_inp=10
        ),
        "IIDDataModel": algo_dict.encode_params(
            "iid", is_verbose=is_verbose, 
            is_test=is_test, dist="gaussian"
        ),
        "ARIMAModel": algo_dict.encode_params(
            "arima", is_verbose=is_verbose, 
            is_test=is_test, order=(2, 1, 5)
        ),
    }

    task = gen_experiment.gen_task_cluster(
        multi_task_algo, "metal", modifier, 
        clus_metal_desc=multi_task_desc, 
        clus_time_desc=None,
        algo_config=default_config,
        len_dataset=len_dataset, 
        len_train_show=len_train_show
    )
    output = gen_experiment.run_experiments(task, save_path=save_path)

    if is_full_result:
        return output
    else:
        dump_json(f"{save_path}all_data.json", output) 

        summary = {}

        for algo_name, result in output.items():
            data = {}
            for metric, r in result.items():
                mean_across_task = np.mean(np.array(r), axis=0)[0]
                data[metric] = mean_across_task
            summary[algo_name] = data 
        
        return summary

def run_hyperparam_search(name, save_folder, multi_task_algo, kernel="Composite_1", is_test=False, is_verbose=False): 
    num_inp = np.arange(2, 12, step=2)
    num_pca = np.arange(2, 7)

    all_results = {
        algo:{m: np.zeros((len(num_inp), len(num_pca))).tolist() for m in metric}
        for algo in multi_task_algo
    }
    all_metal_name = find_all_metal_names()
    
    for i, len_inp in enumerate(num_inp):
        for j, pca in enumerate(num_pca):
            pca_modifier = {
                metal: CompressMethod(int(pca), "pca", info={})
                for metal in all_metal_name
            }
            curr_result = run_multi_task_gp(
                f"{save_folder}/hyper_search_{name}/run_pca_{pca}_feat_{len_inp}/",
                pca_modifier,
                multi_task_desc=[all_metal_name],
                len_inp=int(len_inp), 
                kernel=kernel, 
                is_test=is_test, 
                is_verbose=is_verbose,
                multi_task_algo=multi_task_algo
            )
            for algo in multi_task_algo:
                for met in metric:
                    all_results[algo][met][i][j] = curr_result[algo][met]
    
            dump_json(f"{save_folder}/hyper_search_{name}/final_result_{name}.json", all_results) 

    return all_results 

def general_testing_legacy(is_verbose, is_test):
    all_metal_name = find_all_metal_names()
    no_modifier = {
        metal: CompressMethod(0, "drop")
        for metal in all_metal_name
    }
    pca_modifier = {
        metal: CompressMethod(3, "pca")
        for metal in all_metal_name
    }
    
    all_algo = [
        "GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel", 
        "DeepGPMultiOut", "DSPPMultiOut", "SparseGPIndex", "SparseMaternGraphGP", 
        "DeepGraphMultiOutputGP", "DeepGraphInfoMaxMultiOutputGP",
        "NonlinearMultiTaskGP", "NonlinearMultiTaskGSPP", "DeepGPGraphPropagate", "DeepGPGraphInteract", "DSPPGraphInteract", "DSPPGraphPropagate"
    ] 
    display_name_to_algo = dict(zip(
        all_algo,[
            "Multi-Task Out", "Independent GP", "Multi-Task Index", 
            "Mean", "ARIMA", "Deep GP", "DSPP", 
            "Sparse Multi-Task Index", "Sparse Matern Graph GP", 
            "Deep Graph Multi Output GP", "Deep Graph InfoMax",
            "Non-Linear Multi-Task GP", "Non-Linear Multi-Task DSPP",
            "Deep Gaussian Process Graph", "Deep Gaussian Process Interaction",
            "DSPP Interaction", "DSPP Graph Propagation"
        ],
    ))
    
    # all_algo = [
    #     "IndependentGP", "IIDDataModel", 
    #     "DeepGPMultiOut", "DSPPMultiOut", "SparseGPIndex", 
    #     "SparseMaternGraphGP", "NonlinearMultiTaskGSPP"
    # ] 

    all_algo = [
        # "DeepGraphMultiOutputGP", 
        # "DeepGraphInfoMaxMultiOutputGP",
        # "NonlinearMultiTaskGP", 
        # "NonlinearMultiTaskGSPP", 
        # "DeepGPGraphPropagate", "DeepGPGraphInteract", 
        # "DSPPGraphInteract", 
        "DSPPGraphPropagate"
    ]

    base_multi_task_fast_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel="Matern", 
        optim_iter=optim_iter,
        len_inp=10,
        lr=0.1, 
        graph_path="exp_result/graph_result/hsic_test_graph.npy"
    )

    base_multi_task_v_fast_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel="Matern", 
        optim_iter=optim_iter,
        len_inp=10,
        lr=0.2, 
        graph_path="exp_result/graph_result/hsic_test_graph.npy"
    )

    base_multi_task_slow_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel="Matern", 
        optim_iter=optim_iter,
        len_inp=10,
        lr=0.05, 
        graph_path="exp_result/graph_result/hsic_test_graph.npy"
    )

    base_multi_task_deep = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel="Matern", 
        optim_iter=optim_iter,
        len_inp=10,
        lr=0.01, 
        graph_path="exp_result/graph_result/hsic_test_graph.npy"
    )

    base_multi_task_slower_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel="Matern", 
        optim_iter=500,
        len_inp=10,
        lr=0.005, 
        graph_path="exp_result/graph_result/hsic_test_graph.npy"
    )

    default_config = {
        "GPMultiTaskMultiOut": base_multi_task_fast_lr,
        "GPMultiTaskIndex": base_multi_task_fast_lr,
        "DeepGPMultiOut": base_multi_task_slow_lr,
        "DSPPMultiOut": base_multi_task_slow_lr,
        "SparseGPIndex": base_multi_task_slow_lr,
        "SparseMaternGraphGP": base_multi_task_slow_lr,
        "DeepGraphMultiOutputGP": base_multi_task_slower_lr,
        "DeepGraphInfoMaxMultiOutputGP": base_multi_task_fast_lr,
        # Optim = 4 is enough....
        "NonlinearMultiTaskGP": base_multi_task_deep,
        "NonlinearMultiTaskGSPP": base_multi_task_deep,
        # "DeepGPGraphPropagate": base_multi_task_fast_lr,
        # "DeepGPGraphInteract": base_multi_task_fast_lr,
        # "DSPPGraphInteract": base_multi_task_v_fast_lr,
        # "DSPPGraphPropagate": base_multi_task_v_fast_lr,
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=is_verbose, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=optim_iter,
            len_inp=10
        ),
        "IIDDataModel": algo_dict.encode_params(
            "iid", is_verbose=is_verbose, 
            is_test=is_test, dist="gaussian"
        ),
        "ARIMAModel": algo_dict.encode_params(
            "arima", is_verbose=is_verbose, 
            is_test=is_test, order=(2, 0, 5)
        ),
    }

    def original_test():
    
        all_algo = [
            "GPMultiTaskMultiOut", 
            "IndependentGP", 
            "GPMultiTaskIndex", 
            "IIDDataModel", 
            "ARIMAModel", 
            "DeepGPMultiOut", 
            "DSPPMultiOut", 
            "SparseGPIndex", 
            # "NonlinearMultiTaskGP", 
            # "NonlinearMultiTaskGSPP", 
        ] 

        # (time_al, time_cu, commodity, time_al_feat, time_cu_feat, commodity_feat) = [
        (time_al_feat, time_cu_feat, commodity_feat) = [
            gen_experiment.gen_task_cluster(
                all_algo, type_task, modifier, 
                clus_metal_desc=metal_desc, 
                clus_time_desc=time_desc,
                algo_config=default_config
            )
            for type_task, modifier, metal_desc, time_desc in [
                # ("time", no_modifier, "aluminium", [[22, 44, 66]]),
                # ("time", no_modifier, "copper", [[22, 44, 66]]),
                # ("metal", no_modifier, [["aluminium", "copper"]], None),
                ("time", pca_modifier, "aluminium", [[22, 44, 66]]),
                ("time", pca_modifier, "copper", [[22, 44, 66]]),
                ("metal", pca_modifier, [["aluminium", "copper"]], None),
            ]
        ]
        
        # task_train = [time_al, commodity, time_al_feat, commodity_feat]
        # task_names = ["Price", "Metal", "Price_Feat", "Metal_Feat"]
        
        task_train = [time_al_feat, commodity_feat]
        task_names = ["Price_Feat", "Metal_Feat"]

        def how_to_plot(super_task):
            # plot_latex(
            #     names=[all_algo, all_algo],
            #     results=[super_task["Price"], super_task["Metal"]],
            #     multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
            #     display_name_to_algo=display_name_to_algo
            # )
            # print()
            # print()
            plot_latex(
                names=[all_algo, all_algo],
                results=[super_task["Price_Feat"], super_task["Metal_Feat"]],
                multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
                display_name_to_algo=display_name_to_algo
            )

        return task_train, task_names, how_to_plot
    
    def test_all_metal():
        all_metal_name = find_all_metal_names()

        task = gen_experiment.gen_task_cluster(
            all_algo, "metal", pca_modifier, 
            clus_metal_desc=[all_metal_name], 
            clus_time_desc=None,
            algo_config=default_config
        )

        def how_to_plot(super_task):
            plot_latex(
                names=[all_algo],
                results=[super_task["All Metal"]],
                multi_task_name=[all_metal_name],
                display_name_to_algo=display_name_to_algo
            )

        return [task], ["All Metal"], how_to_plot

    task_train, task_names, how_to_plot = original_test()
    # task_train, task_names, how_to_plot = test_all_metal()

    super_task = {}
    for task, name in zip(task_train, task_names):
        all_out = gen_experiment.run_experiments(task)
        super_task.update({name: all_out})
    
    dump_json("save/all_data.json", super_task) 
    # super_task = load_json("save/all_data.json")

    how_to_plot(super_task)

def general_testing(save_path, setting, is_verbose, is_test):
    print(save_path)
    all_metal_name = find_all_metal_names()

    for i, (curr_algo, len_inp, pca_dim, kernel, graph_path) in enumerate(setting):
        pca_modifier = {
            metal: CompressMethod(int(pca_dim), "pca", info={})
            for metal in all_metal_name
        }

        # try:
        error_results = run_multi_task_gp(
            save_path, pca_modifier, 
            multi_task_desc=[all_metal_name], 
            len_inp=len_inp, 
            len_dataset=default_len_dataset, 
            len_train_show=default_len_train_show, 
            kernel=kernel, 
            is_test=is_test, is_verbose=is_verbose, 
            multi_task_algo=[curr_algo],
            is_full_result=True,
            graph_path=graph_path
        )
        # except RuntimeError: 
        #     error_results = {curr_algo : {"CRPS": None}}

        if i > 0:
            result = load_json(f"{save_path}grid_result.json")
            result.update(error_results)  
            dump_json(f"{save_path}grid_result.json", result)
        else:
            dump_json(f"{save_path}grid_result.json", error_results)

    
    result = load_json(f"{save_path}grid_result.json")
    plot_latex(
        names=[[data[0] for data in setting]],
        results=[result],
        multi_task_name=[all_metal_name],
        display_name_to_algo=algo_dict.class_name_to_display
    )

    pass

def grid_commodities_run(save_path, setting, is_test=False, is_verbose=False):
    all_metal_name = find_all_metal_names()

    num_metal = len(all_metal_name)
    # multi_task_algo = ["GPMultiTaskMultiOut", "GPMultiTaskIndex"]

    counter = 0

    # error_matrix = np.zeros((num_metal, num_metal))
    error_dict = {
        algo: np.zeros((num_metal, num_metal)).tolist()
        for algo, _, _, _ in setting
    }

    for i in range(num_metal):
        for j in range(i, num_metal):
            counter += 1
            if i != j:
                for curr_algo, len_inp, pca_dim, kernel in setting:
                    pca_modifier = {
                        metal: CompressMethod(int(pca_dim), "pca", info={})
                        for metal in all_metal_name
                    }

                    try:
                        error_results = run_multi_task_gp(
                            save_path, pca_modifier, 
                            multi_task_desc=[[all_metal_name[j], all_metal_name[i]]], 
                            len_inp=len_inp, 
                            len_dataset=default_len_dataset, 
                            len_train_show=default_len_train_show, 
                            kernel=kernel, 
                            is_test=is_test, is_verbose=is_verbose, 
                            multi_task_algo=[curr_algo]
                        )
                    except RuntimeError: 
                        error_results = {curr_algo : {"CRPS": None}}

                    error_dict[curr_algo][j][i] = 0 if i == j else error_results[curr_algo]["CRPS"]

                    dump_json(f"{save_path}/grid_result.json", error_dict) 
    
def update_json(path, entry):
    if os.path.isfile(path): 
        updated_version = {}
        current_json = load_json(path)
        for k in entry:
            if k in current_json:
                updated_version[k] = current_json[k].copy()
                updated_version[k].update(entry[k])
            else:
                updated_version[k] = entry[k]
    else:
        updated_version = entry
    
    dump_json(path, updated_version) 


def grid_compare_clusters(setting, cluster_data_path, save_path, is_test=False, is_verbose=False, cluster_names=None):

    cluster_dict = load_json(cluster_data_path)
    all_metal_name = find_all_metal_names()

    all_results = {}

    compare_cluster_path = f"{save_path}/compare_cluster.json"

    for curr_algo, len_inp, pca_dim, kernel in setting:
        pca_modifier = {
            metal: CompressMethod(int(pca_dim), "pca", info={})
            for metal in all_metal_name
        }

        for test_name, cluster_index in cluster_dict.items():

            if not cluster_names is None:
                if not test_name in cluster_names:
                    continue

            curr_save_path = save_path + "/" + "-".join(test_name.split(" ")) + "/"
            try:
                error_results = run_multi_task_gp(
                    curr_save_path, pca_modifier, 
                    multi_task_desc=gen_experiment.cluster_index_to_nested(cluster_index), 
                    len_inp=len_inp, 
                    len_dataset=default_len_dataset, 
                    len_train_show=default_len_train_show, 
                    kernel=kernel, 
                    is_test=is_test, is_verbose=is_verbose, 
                    multi_task_algo=[curr_algo]
                )
            except RuntimeError: 
                error_results = {curr_algo : {"CRPS": None}}

            all_results.update({
                test_name: {
                    k: v["CRPS"]
                    for k, v in error_results.items()
                }
            }) 
            
            update_json(compare_cluster_path, all_results)


            # Running all
        curr_save_path = save_path + "/full_model/"

        try:
            error_results = run_multi_task_gp(
                curr_save_path, pca_modifier, 
                multi_task_desc=[all_metal_name], 
                len_inp=len_inp, 
                len_dataset=default_len_dataset, 
                len_train_show=default_len_train_show, 
                kernel=kernel, 
                is_test=is_test, is_verbose=is_verbose, 
                multi_task_algo=[curr_algo]
            )
        except RuntimeError: 
            error_results = {curr_algo : {"CRPS": None}}

        all_results.update({
            "full_model": {
                k: v["CRPS"]
                for k, v in error_results.items()
            }
        })

        update_json(compare_cluster_path, all_results)

def run_ARMA_param_search(save_folder="exp_result/hyper_param_arma_test"):
    metal_names = find_all_metal_names()
    all_output_data = [
        get_data(metal, is_price_only=False, is_feat=False)
        for metal in metal_names
    ]
    test = all_output_data[0]["Price"]

    
    result = {
        metal: np.zeros((10, 10, 10))
        for metal in metal_names
    }

    create_folder(save_folder)

    for metal in metal_names:
        for i, s1 in enumerate(np.arange(2, 12, step=1)):
            for j, s2 in enumerate(np.arange(2, 12, step=1)):
                for k, s3 in enumerate(np.arange(2, 12, step=1)):
                    print(f"At {i}, {j} and {k} with metal {metal}")
                    try:
                        model = ARIMA(test, order=(s1, s3, s2))
                        model_fit = model.fit(method="innovations_mle")
                        result[metal][i, j, k] = model_fit.aic 
                    except ValueError:
                        print(f"Value Error At {s1}, {s2}")
                        result[metal][i, j, k] = 10 

                    np.save(f"{save_folder}/{metal}.npy", result[metal])

def grid_compare_graph_run(save_path, setting, is_test, is_verbose):
    graph_path = [
        "exp_result/graph_result/distance correlation_test_graph.npy",
        "exp_result/graph_result/hsic_test_graph.npy",
        "exp_result/graph_result/kendell_test_graph.npy",
        "exp_result/graph_result/peason_test_graph.npy",
        "exp_result/graph_result/spearman_test_graph.npy"
    ]

    graph_path = graph_path[:2] if is_test else graph_path

    def get_graph_name(name):
        return name.split("/")[-1].split("_")[0]
    
    get_graph_name(graph_path[0])
        
    base_line_algo = ["SparseGPIndex", "DeepGPMultiOut"]
    all_metal_name = find_all_metal_names()
            

    all_results = {}
    save_graph_path = f"{save_path}/compare_graph.json"

    for curr_algo, len_inp, pca_dim, kernel in setting:
        pca_modifier = {
            metal: CompressMethod(int(pca_dim), "pca", info={})
            for metal in all_metal_name
        }

        for curr_graph_path in graph_path:
            test_name = get_graph_name(curr_graph_path)
            curr_save_path = save_path + test_name + "/"

            try:
                error_results = run_multi_task_gp(
                    curr_save_path, pca_modifier, multi_task_desc=[all_metal_name], 
                    len_inp=len_inp, 
                    len_dataset=default_len_dataset, 
                    len_train_show=default_len_train_show, 
                    kernel=kernel, 
                    graph_path=curr_graph_path, 
                    is_test=is_test, is_verbose=is_verbose, 
                    multi_task_algo=[curr_algo]
                )
            except RuntimeError: 
                error_results = {curr_algo : {"CRPS": None}}

            all_results.update({
                test_name : {
                    k: v["CRPS"]
                    for k, v in error_results.items()
                }
            })
            
            update_json(save_graph_path, all_results)

    curr_save_path = save_path + "full_model/"

    try:
        error_results = run_multi_task_gp(
            curr_save_path, pca_modifier, multi_task_desc=[all_metal_name], 
            len_inp=len_inp, 
            len_dataset=default_len_dataset, 
            len_train_show=default_len_train_show, 
            kernel="RBF", 
            graph_path=None, 
            is_test=is_test, is_verbose=is_verbose, 
            multi_task_algo=base_line_algo
        )
    except RuntimeError: 
        error_results = {curr_algo : {"CRPS": None}}

    all_results.update({
        "no_graph_model" : {
            k: v["CRPS"]
            for k, v in error_results.items()
        }
    })
        
    update_json(save_graph_path, all_results)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Are we testing", dest='is_test', action='store_true')
    parser.add_argument("--train", help="Are we training", dest='is_test', action='store_false')
    parser.set_defaults(is_test=True)

    parser.add_argument("--not-verbose", dest='is_verbose', action='store_false')
    parser.add_argument("--verbose", help="Are we training", dest='is_verbose', action='store_true')
    parser.set_defaults(is_verbose=False)

    args = parser.parse_args()
    return args

def hyperparameter_search():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose
    
    create_folder("save")
    
    # multi_task_algo=["GPMultiTaskIndex", "GPMultiTaskMultiOut"],

    # run_hyperparam_search(
    #     "matern", "exp_result/save_hyper", 
    #     multi_task_algo=["SparseGPIndex", "GPMultiTaskMultiOut", "GPMultiTaskIndex"],
    #     kernel="Matern", is_test=is_test, 
    #     is_verbose=is_verbose
    # )
    # run_hyperparam_search(
    #     "rbf", "exp_result/save_hyper", 
    #     multi_task_algo=["SparseGPIndex", "GPMultiTaskMultiOut", "GPMultiTaskIndex"],
    #     kernel="RBF", is_test=is_test, 
    #     is_verbose=is_verbose
    # )
    
    # run_hyperparam_search(
    #     "matern", "exp_result/save_hyper_deep_dspp_matern", 
    #     multi_task_algo=["DSPPMultiOut"],
    #     # multi_task_algo=["DSPPMultiOut", "NonlinearMultiTaskGSPP"],
    #     kernel="Matern", is_test=is_test, 
    #     is_verbose=is_verbose
    # )
    # run_hyperparam_search(
    #     "rbf", "exp_result/save_hyper_dspp_rbf", 
    #     multi_task_algo=["DSPPMultiOut", "NonlinearMultiTaskGSPP"],
    #     kernel="RBF", is_test=is_test, 
    #     is_verbose=is_verbose
    # )
    
    # run_hyperparam_search(
    #     "matern", "exp_result/save_hyper_deep_2", 
    #     multi_task_algo=["DeepGPMultiOut", "NonlinearMultiTaskGP"],
    #     kernel="Matern", is_test=is_test, 
    #     is_verbose=is_verbose
    # )
    # run_hyperparam_search(
    #     "rbf", "exp_result/save_hyper_deep_2", 
    #     multi_task_algo=["DeepGPMultiOut", "NonlinearMultiTaskGP"],
    #     kernel="RBF", is_test=is_test, 
    #     is_verbose=is_verbose
    # )

def general_test_run():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose
    
    best_setting = [
        ("IIDDataModel", 
            2, 2, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"),
        ("ARIMAModel", 
            2, 2, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"),
        ("IndependentGP", 
            2, 2, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"),
        ("SparseGPIndex", 
            2, 2, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"),
        ("GPMultiTaskMultiOut", 
            2, 2, 
            "Matern", "exp_result/graph_result/distance correlation_test_graph.npy"), 
        ("GPMultiTaskIndex", 
            8, 2, "RBF", 
            "exp_result/graph_result/distance correlation_test_graph.npy"),
        ("DeepGPMultiOut", 
            10, 6, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"), 
        ("DSPPMultiOut", 
            10, 6, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"), 
        ("NonlinearMultiTaskGP", 
            6, 3, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"), 
        ("NonlinearMultiTaskGSPP", 
            6, 3, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"), 
        ("DeepGraphMultiOutputGP", 
            2, 2, "Matern", 
            "exp_result/graph_result/spearman_test_graph.npy"), 
        ("DeepGraphInfoMaxMultiOutputGP",
            2, 2, "Matern", 
            "exp_result/graph_result/hsic_test_graph.npy"),
        ("SparseMaternGraphGP", 
            2, 2, "Matern", 
            "exp_result/graph_result/distance correlation_test_graph.npy"),
        ("DeepGPGraphPropagate", 
            3, 10, "Matern", 
            "exp_result/graph_result/hsic_test_graph.npy"),
        ("DeepGPGraphInteract", 
            3, 10, "Matern", 
            "exp_result/graph_result/hsic_test_graph.npy"),
        ("DSPPGraphInteract", 
            3, 10, "Matern", 
            "exp_result/graph_result/hsic_test_graph.npy"),
        ("DSPPGraphPropagate",
            3, 10, "Matern", 
            "exp_result/graph_result/hsic_test_graph.npy"),
    ] 

    save_path = "exp_result/general_running/"
    # general_testing(save_path, best_setting, is_verbose, is_test)
    general_testing_legacy(is_verbose, is_test)

def compare_cluster():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose

    best_setting = [
        # ("SparseGPIndex", 2, 2, "Matern"),
        # ("GPMultiTaskMultiOut", 2, 2, "Matern"), 
        # ("GPMultiTaskIndex", 8, 2, "RBF"),
        ("DeepGPMultiOut", 10, 6, "Matern"), 
        ("DSPPMultiOut", 10, 6, "Matern"), 
        ("NonlinearMultiTaskGP", 6, 3, "RBF"), 
        ("NonlinearMultiTaskGSPP", 6, 3, "RBF"), 
    ] 
    
    second_best_setting = [
        ("SparseGPIndex", 2, 3, "Matern"),
        ("GPMultiTaskMultiOut", 2, 4, "Matern"), 
        ("GPMultiTaskIndex", 10, 2, "RBF"),
        ("DeepGPMultiOut", 10, 6, "RBF"), 
        ("DSPPMultiOut", 10, 6, "RBF"), 
        ("NonlinearMultiTaskGP", 6, 3, "Matern"), 
        ("NonlinearMultiTaskGSPP", 6, 3, "Matern"),
    ] 
    
    worst_setting = [
        ("SparseGPIndex", 4, 3, "RBF"),
        ("GPMultiTaskMultiOut", 10, 6, "RBF"), 
        ("GPMultiTaskIndex", 2, 3, "RBF"),
        ("DeepGPMultiOut", 8, 2, "Matern"), 
        ("DSPPMultiOut", 8, 2, "Matern"), 
        ("NonlinearMultiTaskGP", 10, 4, "RBF"), 
        ("NonlinearMultiTaskGSPP", 10, 4, "RBF"), 
    ] 

    grid_compare_clusters(
        best_setting,
        "exp_result/cluster_result/feat_data/cluster_4.json", 
        "exp_result/cluster_compare", 
        is_test=is_test, is_verbose=is_verbose
    )

def compare_many_clusters():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose

    best_setting = [
        # ("SparseGPIndex", 2, 2, "Matern"),
        # ("GPMultiTaskMultiOut", 2, 2, "Matern"), 
        # ("GPMultiTaskIndex", 8, 2, "RBF"),
        # ("DeepGPMultiOut", 10, 6, "Matern"), 
        # ("DSPPMultiOut", 10, 6, "Matern"), 
        # ("NonlinearMultiTaskGP", 6, 3, "Matern"), 
        ("NonlinearMultiTaskGSPP", 6, 3, "Matern"), 
    ] 
    
    worst_setting = [
        # ("SparseGPIndex", 4, 3, "RBF"),
        # ("GPMultiTaskMultiOut", 10, 6, "RBF"), 
        # ("GPMultiTaskIndex", 2, 3, "RBF"),
        # ("DeepGPMultiOut", 8, 2, "Matern"), 
        # ("DSPPMultiOut", 8, 2, "Matern"), 
        ("NonlinearMultiTaskGP", 10, 4, "RBF"), 
        # ("NonlinearMultiTaskGSPP", 10, 4, "Matern"), 
    ] 

    save_folder = "exp_result/range_cluster_test"

    num_cluster = [2, 3, 4, 5, 6, 7]
    create_folder(save_folder)
    
    missing_entries_best = {
        2: ["euclidean", "euclidean knn", "dtw knn"],
        # 3: ["soft-dtw divergence"],
        # 4: ["euclidean knn", "dtw knn", "softdtw knn", "expert"],
        # 5: ["peason", "softdtw knn"],
        # 6: ["euclidean"],
        # 7: ["dtw knn"]
    }

    missing_entries_worst = {
        2: ["dtw knn"],
        4: ["kendel"]
    }

    for n in num_cluster: 
        if not n in missing_entries_best:
            continue

        grid_compare_clusters(
            best_setting,
            f"exp_result/cluster_result/feat_data/cluster_{n}.json", 
            f"{save_folder}/cluster_compare_{n}", 
            is_test=is_test, is_verbose=is_verbose, 
            cluster_names=missing_entries_best[n]
        )

def grid_commodities():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose
    
    best_setting = [
        ("SparseGPIndex", 2, 2, "Matern"),
        ("GPMultiTaskMultiOut", 2, 2, "Matern"), 
        ("GPMultiTaskIndex", 8, 2, "RBF"),
        # ("DeepGPMultiOut", 10, 6, "Matern"), 
        # ("DSPPMultiOut", 10, 6, "Matern"), 
        # ("NonlinearMultiTaskGP", 6, 3, "Matern"), 
        # ("NonlinearMultiTaskGSPP", 6, 3, "Matern"), 
    ] 
    
    grid_commodities_run(
        "exp_result/grid_corr_plot/", 
        best_setting,
        is_test=is_test, is_verbose=is_verbose
    )
    # plot_grid_commodity("grid_corr_plot/grid_result.json")

def grid_compare_graph():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose
    
    best_graph_setting = [
        ("DeepGraphMultiOutputGP", 2, 2, "Matern"), 
        ("DeepGraphInfoMaxMultiOutputGP", 2, 2, "Matern"),
        ("SparseMaternGraphGP", 2, 2, "Matern"),
    ] 
    
    worst_graph_setting = [
        ("DeepGraphMultiOutputGP", 10, 6, "RBF"), 
        ("DeepGraphInfoMaxMultiOutputGP", 10, 6, "RBF"), 
        ("SparseMaternGraphGP", 4, 3, "RBF"),
    ] 
    
    grid_compare_graph_run(
        "exp_result/graph_compare/",
        worst_graph_setting,
        is_test=is_test, is_verbose=is_verbose
    )


def main():
    # compare_cluster()
    # compare_many_clusters()
    # hyperparameter_search()
    run_ARMA_param_search()
    # general_test_run()
    # grid_commodities()
    # grid_compare_graph()


if __name__ == '__main__':
    main()

