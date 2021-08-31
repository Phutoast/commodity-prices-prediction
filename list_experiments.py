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
optim_iter=25

def run_multi_task_gp(save_path, modifier, multi_task_desc, len_inp=10, 
    len_dataset=default_len_dataset, len_train_show=default_len_train_show, 
    kernel="Composite_1", 
    graph_path="exp_result/graph_result/distance correlation_test_graph.npy", 
    is_test=False, is_verbose=False, 
    multi_task_algo=multi_task_algo
):

    config = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel=kernel, 
        optim_iter=optim_iter,
        len_inp=len_inp,
        lr=0.05,
        graph_path=graph_path
    )
    multi_task_config = {
        "GPMultiTaskMultiOut": config,
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=is_verbose, 
            is_test=is_test, 
            kernel=kernel, 
            optim_iter=optim_iter,
            len_inp=len_inp,
        ),
        "GPMultiTaskIndex": config,
        "DeepGPMultiOut": config,
        "DSPPMultiOut": config,
        "SparseGPIndex": config,
        "SparseMaternGraphGP": config
    }
    task = gen_experiment.gen_task_cluster(
        multi_task_algo, "metal", modifier, 
        clus_metal_desc=multi_task_desc, 
        clus_time_desc=None,
        algo_config=multi_task_config,
        len_dataset=len_dataset, 
        len_train_show=len_train_show
    )
    output = gen_experiment.run_experiments(task, save_path=save_path)
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

def general_testing(is_verbose, is_test):
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
        "DeepGraphInfoMaxMultiOutputGP",
        # "NonlinearMultiTaskGP", "NonlinearMultiTaskGSPP", "DeepGPGraphPropagate", "DeepGPGraphInteract", "DSPPGraphInteract", "DSPPGraphPropagate"
    ]

    # all_algo = ["DeepGraphMultiOutputGP", "DeepGraphInfoMaxMultiOutputGP"]
    # all_algo = ["DeepGraphMultiOutputGP"]
    # all_algo = ["GPMultiTaskIndex"]
    # all_algo = ["DeepGPMultiOut"]
    # all_algo = ["DSPPMultiOut"]
    # all_algo = ["GPMultiTaskIndex"]
    # all_algo = ["NonlinearMultiTaskGP"]


    base_multi_task_fast_lr = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel="Matern", 
        optim_iter=optim_iter,
        len_inp=10,
        lr=0.1, 
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
        "DeepGraphInfoMaxMultiOutputGP": base_multi_task_slow_lr,
        "NonlinearMultiTaskGP": base_multi_task_slower_lr,
        "NonlinearMultiTaskGSPP": base_multi_task_slower_lr,
        "DeepGPGraphPropagate": base_multi_task_fast_lr,
        "DeepGPGraphInteract": base_multi_task_fast_lr,
        "DSPPGraphInteract": base_multi_task_fast_lr,
        "DSPPGraphPropagate": base_multi_task_fast_lr,
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

        (time_al, time_cu, commodity, time_al_feat, time_cu_feat, commodity_feat) = [
            gen_experiment.gen_task_cluster(
                all_algo, type_task, modifier, 
                clus_metal_desc=metal_desc, 
                clus_time_desc=time_desc,
                algo_config=default_config
            )
            for type_task, modifier, metal_desc, time_desc in [
                ("time", no_modifier, "aluminium", [[22, 44, 66]]),
                ("time", no_modifier, "copper", [[22, 44, 66]]),
                ("metal", no_modifier, [["aluminium", "copper"]], None),
                ("time", pca_modifier, "aluminium", [[22, 44, 66]]),
                ("time", pca_modifier, "copper", [[22, 44, 66]]),
                ("metal", pca_modifier, [["aluminium", "copper"]], None),
            ]
        ]
        
        task_train = [time_al, commodity, time_al_feat, commodity_feat]
        task_names = ["Price", "Metal", "Price_Feat", "Metal_Feat"]

        def how_to_plot(super_task):
            plot_latex(
                names=[all_algo, all_algo],
                results=[super_task["Price"], super_task["Metal"]],
                multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
                display_name_to_algo=display_name_to_algo
            )
            print()
            print()
            plot_latex(
                names=[all_algo, all_algo],
                results=[super_task["Price_Feat"], super_task["Metal_Feat"]],
                multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
                display_name_to_algo=display_name_to_algo
            )

        return task_train, task_names, how_to_plot
    
    def test_all_metal():
        task = gen_experiment.gen_task_cluster(
            all_algo, "metal", pca_modifier, 
            clus_metal_desc=[find_all_metal_names()], 
            clus_time_desc=None,
            algo_config=default_config
        )

        def how_to_plot(super_task):
            plot_latex(
                names=[all_algo],
                results=[super_task["All Metal"]],
                multi_task_name=[find_all_metal_names()],
                display_name_to_algo=display_name_to_algo
            )

        return [task], ["All Metal"], how_to_plot

    # task_train, task_names, how_to_plot = original_test()
    task_train, task_names, how_to_plot = test_all_metal()

    super_task = {}
    for task, name in zip(task_train, task_names):
        all_out = gen_experiment.run_experiments(task)
        super_task.update({name: all_out})
    
    dump_json("save/all_data.json", super_task) 
    # super_task = load_json("save/all_data.json")

    how_to_plot(super_task)

def grid_commodities_run(save_path, len_inp, pca_dim, kernel, is_test=False, is_verbose=False):
    all_metal_name = find_all_metal_names()

    num_metal = len(all_metal_name)
    multi_task_algo = ["GPMultiTaskMultiOut", "GPMultiTaskIndex"]

    pca_modifier = {
        metal: CompressMethod(int(pca_dim), "pca", info={})
        for metal in all_metal_name
    }

    counter = 0

    # error_matrix = np.zeros((num_metal, num_metal))
    error_dict = {
        algo: np.zeros((num_metal, num_metal))
        for algo in multi_task_algo
    }

    for i in range(num_metal):
        for j in range(i, num_metal):
            counter += 1
            if i != j:
                error_results = run_multi_task_gp(
                    save_path, pca_modifier, 
                    multi_task_desc=[[all_metal_name[j], all_metal_name[i]]], 
                    len_inp=len_inp, 
                    len_dataset=default_len_dataset, 
                    len_train_show=default_len_train_show, 
                    kernel=kernel, 
                    is_test=is_test, is_verbose=is_verbose, 
                    multi_task_algo=multi_task_algo
                )

            for algo in multi_task_algo:
                error_dict[algo][j, i] = 0 if i == j else error_results[algo]["CRPS"]
    
    for algo in multi_task_algo:
        error_dict[algo] = (error_dict[algo].T + error_dict[algo]).tolist()
    
    error_dict["metal_names"] = [
        metal_desc.metal_to_display_name[name]
        for name in all_metal_name
    ]
    
    dump_json(f"{save_path}/grid_result.json", error_dict) 

def grid_compare_clusters(cluster_data_path, save_path, list_len_inp, 
    list_pca_dim, kernel, is_test=False, is_verbose=False):

    cluster_dict = load_json(cluster_data_path)
    all_metal_name = find_all_metal_names()

    all_results = {}
    list_algo = [
        "GPMultiTaskMultiOut", "GPMultiTaskIndex", 
        "DeepGPMultiOut", "DSPPMultiOut", 
        # "SparseGPIndex"
    ]
    compare_cluster_path = f"{save_path}/compare_cluster.json"

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

    for len_inp, pca_dim, curr_algo in zip(list_len_inp, list_pca_dim, list_algo):
        pca_modifier = {
            metal: CompressMethod(int(pca_dim), "pca", info={})
            for metal in all_metal_name
        }

        for test_name, cluster_index in cluster_dict.items():

            curr_save_path = save_path + "/" + "-".join(test_name.split(" ")) + "/"
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

            all_results.update({
                test_name: {
                    k: v["CRPS"]
                    for k, v in error_results.items()
                }
            }) 
            
            update_json(compare_cluster_path, all_results)


            # Running all
        curr_save_path = save_path + "/full_model/"
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

        all_results.update({
            "full_model": {
                k: v["CRPS"]
                for k, v in error_results.items()
            }
        })

        update_json(compare_cluster_path, all_results)

def run_ARMA_param_search(save_folder="exp_result/hyper_param_arma"):
    metal_names = find_all_metal_names()
    all_output_data = [
        get_data(metal, is_price_only=False, is_feat=False)
        for metal in metal_names
    ]
    test = all_output_data[0]["Price"]

    
    result = {
        metal: np.zeros((10, 11))
        for metal in metal_names
    }

    create_folder(save_folder)

    for metal in metal_names:
        for i, s1 in enumerate(np.arange(2, 12, step=1)):
            for j, s2 in enumerate(np.arange(2, 12, step=1)):
                print(f"At {i} and {j} with metal {metal}")
                try:
                    model = ARIMA(test, order=(s1, 1, s2))
                    model_fit = model.fit(method="innovations_mle")
                    result[metal][i, j] = model_fit.aic 
                except ValueError:
                    print(f"Value Error At {s1}, {s2}")
                    result[metal][i, j] = 10 

                np.save(f"{save_folder}/{metal}.npy", result[metal])

def grid_compare_graph_run(save_path, len_inp, pca_dim, is_test, is_verbose):
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
        
    list_algo = ["SparseMaternGraphGP", 
        "DeepGraphMultiOutputGP", "DeepGraphInfoMaxMultiOutputGP"
    ]
    base_line_algo = ["SparseGPIndex", "DeepGPMultiOut"]

    all_metal_name = find_all_metal_names()
            
    pca_modifier = {
        metal: CompressMethod(int(pca_dim), "pca", info={})
        for metal in all_metal_name
    }

    all_results = {}

    for curr_graph_path in graph_path:
        test_name = get_graph_name(curr_graph_path)
        curr_save_path = save_path + test_name + "/"

        error_results = run_multi_task_gp(
            curr_save_path, pca_modifier, multi_task_desc=[all_metal_name], 
            len_inp=len_inp, 
            len_dataset=default_len_dataset, 
            len_train_show=default_len_train_show, 
            kernel="RBF", 
            graph_path=curr_graph_path, 
            is_test=is_test, is_verbose=is_verbose, 
            multi_task_algo=list_algo
        )

        all_results.update({
            test_name : {
                k: v["CRPS"]
                for k, v in error_results.items()
            }
        })
        
        dump_json(f"{save_path}/compare_graph.json", all_results) 

    curr_save_path = save_path + "full_model/"
    no_graph_results = run_multi_task_gp(
        curr_save_path, pca_modifier, multi_task_desc=[all_metal_name], 
        len_inp=len_inp, 
        len_dataset=default_len_dataset, 
        len_train_show=default_len_train_show, 
        kernel="RBF", 
        graph_path=None, 
        is_test=is_test, is_verbose=is_verbose, 
        multi_task_algo=base_line_algo
    )

    all_results.update({
        "no_graph_model" : {
            k: v["CRPS"]
            for k, v in no_graph_results.items()
        }
    })
    
    dump_json(f"{save_path}/compare_graph.json", all_results) 

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

    run_hyperparam_search(
        "matern", "exp_result/save_hyper", 
        multi_task_algo=["SparseGPIndex", "GPMultiTaskMultiOut", "GPMultiTaskIndex"],
        kernel="Matern", is_test=is_test, 
        is_verbose=is_verbose
    )
    run_hyperparam_search(
        "rbf", "exp_result/save_hyper", 
        multi_task_algo=["SparseGPIndex", "GPMultiTaskMultiOut", "GPMultiTaskIndex"],
        kernel="RBF", is_test=is_test, 
        is_verbose=is_verbose
    )
    # run_hyperparam_search(
    #     "matern_periodic", "exp_result/save_hyper", 
    #     multi_task_algo=["SparseGPIndex", "GPMultiTaskMultiOut"],
    #     kernel="Composite_1", is_test=is_test, 
    #     is_verbose=is_verbose
    # )
    # run_hyperparam_search(
    #     "rbf_periodic", "exp_result/save_hyper", 
    #     multi_task_algo=["SparseGPIndex", "GPMultiTaskMultiOut"],
    #     kernel="Composite_2", is_test=is_test, 
    #     is_verbose=is_verbose
    # )

def general_test_run():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose
    general_testing(is_verbose, is_test)

def compare_cluster():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose

    # grid_compare_clusters(
    #     "result/cluster_result/feat_data/cluster_4.json", 
    #     "exp_result/cluster_compare", 
    #     3, 3, "Composite_1", is_test=is_test, is_verbose=is_verbose
    # )
    
    grid_compare_clusters(
        "exp_result/cluster_result/feat_data/cluster_4.json", 
        "exp_result/cluster_compare", 
        [3]*6, [5]*6, "RBF", is_test=is_test, is_verbose=is_verbose
    )

def grid_commodities():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose
    
    grid_commodities_run(
        "exp_result/grid_corr_plot/", 
        5, 5, "RBF", 
        is_test=is_test, is_verbose=is_verbose
    )
    plot_grid_commodity("grid_corr_plot/grid_result.json")

def grid_compare_graph():
    args = argument_parser()
    is_test=args.is_test
    is_verbose=args.is_verbose
    
    grid_compare_graph_run(
        "exp_result/grid_graph/",
        len_inp=10, pca_dim=3,
        is_test=is_test, is_verbose=is_verbose
    )

def main():
    # compare_cluster()
    # hyperparameter_search()
    # run_ARMA_param_search()
    general_test_run()
    # grid_commodities()
    # grid_compare_graph()


if __name__ == '__main__':
    main()

