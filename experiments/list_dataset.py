from utils.data_structure import DatasetTaskDesc, CompressMethod

length_dataset = 795

def gen_datasets(type_task="time", modifier=None, metal_type="aluminium"):

    def cal_modifier_feature(inp_metal_list):
        metal_modifier, additional_features = [], []
        for metal in inp_metal_list:
            curr_modifier = modifier[metal]
            metal_modifier.append(curr_modifier)

            if not curr_modifier is None:
                dim = curr_modifier.compress_dim
                additional_features += [f"{metal}.Feature{i+1}" for i in range(dim)]
        return metal_modifier, additional_features
        
    if modifier is None:
        additional_features = []
        modifier = {"aluminium": None, "copper": None}
    elif not isinstance(modifier, dict):
        raise TypeError("Modifier has to be passed as Dict")

    if type_task == "time": 
        inp_metal_list = [metal_type]
        metal_modifier, additional_features = cal_modifier_feature(inp_metal_list)
        all_time_step = [22, 44, 66]

        dataset = [
            DatasetTaskDesc(
                inp_metal_list=inp_metal_list,
                metal_modifier=metal_modifier,
                use_feature=["Date"] + additional_features,
                use_feat_tran_lag=None,
                out_feature=f"{metal_type}.Price",
                out_feat_tran_lag=(time, 0, "id"),
                len_dataset=length_dataset,
            )
            for time in all_time_step
        ]
        return dataset

    elif type_task == "metal":
        all_metals = ["aluminium", "copper"] 

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
                len_dataset=length_dataset,
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
                len_dataset=length_dataset,
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
                len_dataset=length_dataset,
            )
            for metal, (modi_list, addi_feature) in zip(all_metals[1:], list_compose[1:])
        ]
        return dataset, use_first_dataset 