import pandas as pd
import datetime
import numpy as np

fake_price_data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
fake_first_day = "2001-01-01"

def generate_fake_data(size=26):
    start_day = datetime.datetime(2001, 1, 1)
    all_days = [
        (start_day + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(size)
    ]
    feature_1 = [
        f"feature_1_day_{(i+1):02}"
        for i in range(size)
    ]
    feature_2 = [
        f"feature_2_day_{(i+1):02}"
        for i in range(size)
    ]
    prices = [np.exp(i) for i in fake_price_data]

    d = {"Date": all_days, "Feature1": feature_1, "Feature2": feature_2, "Price": prices}
    return pd.DataFrame(data=d)


if __name__ == '__main__':
    print(generate_fake_data())
