from experiments import algo_dict
from examples.simple_example import example_plot_walk_forward

def create_exp(dataset, algo, algo_config):
    num_task = len(dataset)

    if algo in algo_dict.class_name:
        train_algo = "IndependentMultiModel"
    else:
        train_algo = algo
    
    using_first = False
    if algo in algo_dict.using_first_algo:
        using_first = True

    exp_setting = {
        "task": {
            "sub_model": [algo_config[algo]] * num_task,
            "dataset": dataset,
            "len_pred_show": 130,
            "len_train_show": (275, 130)
        }, "algo": train_algo, 
        "using_first": using_first
    }
    return exp_setting

def run_experiments(exp_list):
    all_results = {}
    for name, exp in exp_list:
        out_result = example_plot_walk_forward(exp, name,
            is_save=True, is_load=False, is_show=False,
            load_path=name, 
        )
 
        result = {
            "MSE": out_result[0],
            "CRPS": out_result[1]
        }
        all_results.update({name: result})
    
    return all_results

def load_experiments(path_exp_len):
    all_results = []
    for name, path in path_exp_len:
        print("Name:", name)
        out_result = example_plot_walk_forward(None, name,
            is_save=False, is_load=True, is_show=False,
            load_path=path, 
        )
        all_results.append(out_result)

    return all_results