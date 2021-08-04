import pandas as pd
import datetime
import numpy as np

from utils.data_preprocessing import identity_modifier

np.random.seed(48)

fake_price_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
fake_feature_data= [62, 40, 33, 97, 90, 16, 69, 63, 4, 98, 49, 68, 75, 42, 3, 47, 1, 5, 71, 58, 93, 56, 92, 13, 76, 73]
fake_first_day = "2001-01-01"
nan_mask = np.array(
[[1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
 [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
 [1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
 [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
 [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
 [1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
 [0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
 [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
 [1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
 [1, 0, 0, 1, 1, 1, 0, 1, 1, 1],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
 [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
 [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
 [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
 [1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
 [1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
 [0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
 [1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
 [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
 [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]]
).astype(np.float32)
nan_mask[nan_mask == 0] = np.nan

def get_loc_non_nan(num_metal, index_list):
    feature_mask = nan_mask[:, index_list]
    all_index = np.nonzero(~np.isnan(feature_mask).any(axis=1))
    return all_index[0]


def generate_fake_data(metal_name, global_modifier=identity_modifier, size=26, is_weird=False, is_nan=False):
    start_day = datetime.datetime(2001, 1, 1)
    all_days = [
        (start_day + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(size)
    ]

    num_metal = int(metal_name[-1])
    prices = [np.exp(i*num_metal) for i in fake_price_data]
    def generate_feat(name):
        return [
            # f"feat_{name}_day_{(i+1):02}"
            np.sqrt(fake_feature_data[i]*num_metal + name)
            for i in range(size)
        ]
    
    if is_nan:
        mask_data = nan_mask
    else:
        mask_data = np.ones_like(nan_mask)
    
    other_feature = {
        f"FeatureFamily.Feature{i}" : np.array(generate_feat(i)) * mask_data[:, i]
        for i in list(range(num_metal, num_metal*2+1))
    }

    d = {
        "Date": all_days, 
    }
    d.update(other_feature)
    d.update({"Price": prices})

    df = pd.DataFrame(data=d)
    if is_weird and metal_name=="metal2":
        return df.reindex([i for i in range(size) if i%2 == 0])

    return df

if __name__ == '__main__':
    print(generate_fake_data(metal_name="metal1", is_nan=True))
    # get_loc_non_nan(1)
