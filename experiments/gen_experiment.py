from experiments import algo_dict
from examples.simple_example import example_plot_walk_forward
from experiments.algo_dict import using_out_only, using_first_algo
from utils.others import find_all_metal_names
from utils.data_structure import DatasetTaskDesc, CompressMethod
from utils.others import find_all_metal_names

def create_exp(dataset, algo, algo_config, len_train_show=(275, 130)):
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
            "len_train_show": len_train_show,
        }, "algo": train_algo, 
        "using_first": using_first
    }
    return exp_setting

def gen_datasets(type_task="time", modifier=None, 
    metal_type="aluminium", len_dataset=794, all_time_step = [22, 44, 66]):

    def cal_modifier_feature(inp_metal_list):
        metal_modifier, additional_features = [], []
        for metal in inp_metal_list:
            curr_modifier = modifier[metal]
            metal_modifier.append(curr_modifier)

            if not curr_modifier is None:
                dim = curr_modifier.compress_dim
                additional_features += [f"{metal}.Feature{i+1}" for i in range(dim)]

        return metal_modifier, additional_features

    all_metals = find_all_metal_names()
        
    if modifier is None:
        additional_features = []
        modifier = {
            metal: None
            for metal in all_metals
        }
    elif not isinstance(modifier, dict):
        raise TypeError("Modifier has to be passed as Dict")
    
    if type_task == "time": 
        assert isinstance(metal_type, str)
        inp_metal_list = [metal_type]
        metal_modifier, additional_features = cal_modifier_feature(inp_metal_list)

        dataset = [
            DatasetTaskDesc(
                inp_metal_list=inp_metal_list,
                metal_modifier=metal_modifier,
                use_feature=["Date"] + additional_features,
                use_feat_tran_lag=None,
                out_feature=f"{metal_type}.Price",
                out_feat_tran_lag=(time, 0, "id"),
                len_dataset=len_dataset,
            )
            for time in all_time_step
        ]
        return dataset

    elif type_task == "metal":
        assert isinstance(metal_type, list) or metal_type is None

        if metal_type is not None:
            all_metals = metal_type
        
        list_compose = [
            cal_modifier_feature([metal])
            for metal in all_metals
        ]

        dataset = [
            DatasetTaskDesc(
                inp_metal_list=[metal],
                metal_modifier=modi_list,
                use_feature=["Date"] + addi_feature,
                use_feat_tran_lag=None,
                out_feature=f"{metal}.Price",
                out_feat_tran_lag=(22, 0, "id"),
                len_dataset=len_dataset,
            )
            for metal, (modi_list, addi_feature) in zip(all_metals, list_compose)
        ]

        rest_price = [f"{metal}.Price" for metal in all_metals[1:]]
        use_feat_tran_lag = [None] + [(22, 0, "id") for metal in all_metals[1:]]
        metal_modifier, additional_features = cal_modifier_feature(all_metals)

        use_first_dataset = [
            DatasetTaskDesc(
                inp_metal_list=all_metals,
                metal_modifier=metal_modifier,
                use_feature=["Date"] + rest_price + additional_features,
                use_feat_tran_lag=use_feat_tran_lag + [None] * len(additional_features),
                out_feature=f"{all_metals[0]}.Price",
                out_feat_tran_lag=(22, 0, "id"),
                len_dataset=len_dataset,
            )
        ]

        use_first_dataset += [
            DatasetTaskDesc(
                inp_metal_list=[metal],
                metal_modifier=modi_list,
                use_feature=["Date"] + addi_feature,
                use_feat_tran_lag=None,
                out_feature=f"{metal}.Price",
                out_feat_tran_lag=(22, 0, "id"),
                len_dataset=len_dataset,
            )
            for metal, (modi_list, addi_feature) in zip(all_metals[1:], list_compose[1:])
        ]
        return dataset, use_first_dataset 

def gen_task_cluster(all_algo, type_task, modifier, clus_metal_desc, clus_time_desc,
    algo_config, len_dataset=794, len_train_show=(274, 130)):

    all_result = []
    for curr_algo in all_algo:

        # Can be extended to having multiple algorithms in multiple cluster
        # To make things simplier we won't do it.
        merge_task_algo = {
            "task": {
                "sub_model": [],
                "dataset": [],
                "len_pred_show": len_dataset,
                "len_train_show": len_train_show
            },
            "algo": [],
            "using_first": []
        }

        iterator_now = {
            "metal": clus_metal_desc,
            "time": clus_time_desc,
        }[type_task.lower()]

        for clus in iterator_now:
            if type_task.lower() == "metal":
                clus_task = gen_task_list([curr_algo], type_task, modifier, clus, algo_config)[0][1]
            elif type_task.lower() == "time":
                clus_task = gen_task_list([curr_algo], type_task, modifier, clus_metal_desc, algo_config, all_time_step=clus)[0][1]

            merge_task_algo["task"]["sub_model"].append(clus_task["task"]["sub_model"])
            merge_task_algo["task"]["dataset"].append(clus_task["task"]["dataset"])
            merge_task_algo["algo"].append(clus_task["algo"])
            merge_task_algo["using_first"].append(clus_task["using_first"])

        all_result.append((curr_algo, merge_task_algo))

    return all_result


def gen_task_list(all_algo, type_task, modifier, metal_type, 
    algo_config, len_dataset=794, len_train_show=(274, 130), all_time_step=[22, 44, 66]):

    def gen_list(dataset, algo_list):
        multi_task_dataset = []
        for algo in algo_list:
            if algo in using_out_only:

                get_type_using_out = lambda x: "drop" if x in ["pca", "drop"] else "id"
                new_modi = {
                    k : CompressMethod(0, get_type_using_out(v.method), v.info)
                    for k, v in modifier.items()
                }
                if type_task == "time":
                    dataset = gen_datasets(
                        type_task, 
                        new_modi,
                        metal_type,
                        len_dataset=len_dataset,
                        all_time_step=all_time_step
                    )
                else:
                    dataset = gen_datasets(
                        type_task, 
                        new_modi,
                        metal_type,
                        len_dataset=len_dataset,
                        all_time_step=all_time_step
                    )[0]

            multi_task_dataset.append(
                (algo, create_exp(
                    dataset, algo, algo_config, len_train_show
                ))
            )
        return multi_task_dataset

    if type_task == "time":
        dataset = gen_datasets(
            "time", modifier, metal_type, 
            len_dataset=len_dataset,
            all_time_step=all_time_step
        )
        time_multi_task = gen_list(dataset, all_algo)
        return time_multi_task

    elif type_task == "metal":
        commo, commo_first = gen_datasets(
            "metal", modifier, metal_type, len_dataset=len_dataset
        )
        metal_multi_task = gen_list(
            commo_first, 
            list(filter(lambda x : x in algo_dict.using_first_algo, all_algo))
        )
        metal_multi_task += gen_list(
            commo, 
            list(filter(lambda x : not x in algo_dict.using_first_algo, all_algo))
        )
        return metal_multi_task
    else:
        raise ValueError("There are only 2 tasks for now, time and metal")

def run_experiments(exp_list, save_path="save/"):
    all_results = {}

    if save_path[-1] != "/":
        save_path = save_path + "/"

    for name, exp in exp_list:
        out_result = example_plot_walk_forward(exp, name,
            is_save=True, is_load=False, is_show=False,
            load_path=name, save_path=save_path
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