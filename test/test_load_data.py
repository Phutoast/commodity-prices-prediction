import numpy as np
from utils.data_preprocessing import load_transform_data
from experiments.eval_methods import prepare_dataset
from utils.data_structure import TrainingPoint
from pandas._testing import assert_frame_equal

import unittest
from unittest.mock import patch
from test.test_utils import generate_fake_data, fake_price_data, fake_first_day

class LoadingDataLag(unittest.TestCase): 

    def setUp(self):
        self.len_inp, self.len_out, self.lag_num, self.skip = 3, 2, 6, 5
        self.the_data = generate_fake_data()
        self.the_data["Price"] = np.log(self.the_data["Price"].to_numpy())
        self.the_data["Date_str"] = self.the_data["Date"]
        self.the_data["Date"] = range(26)
    
    def getting_points_data(self, all_true_data):
        points = []
        for data in all_true_data:
            data_temp = []
            for d in data:
                data = d[["Date", "Feature1", "Feature2"]]
                label = d[["Date_str", "Price"]].rename(columns={"Date_str": "Date"})
                data_temp.append(data)
                data_temp.append(label)
            points.append(TrainingPoint(*data_temp))
        return points
    
    def getting_lag_data(self, log_prices, all_true_data):
        points = []
        for inp_data_ind, out_data_ind in all_true_data:
            inp_data = self.the_data.iloc[inp_data_ind]
            out_data = self.the_data.iloc[out_data_ind]

            data_inp = inp_data.copy()[["Date", "Feature1", "Feature2"]]
            label_inp = inp_data.copy()[["Date_str", "Price"]].rename(columns={"Date_str": "Date"})
            label_inp["Price"] = log_prices["Price"].iloc[inp_data_ind]

            data_out = out_data.copy()[["Date", "Feature1", "Feature2"]]
            label_out = out_data.copy()[["Date_str", "Price"]].rename(columns={"Date_str": "Date"})
            label_out["Price"] = log_prices["Price"].iloc[out_data_ind]
            points.append(TrainingPoint(data_inp, label_inp, data_out, label_out))

        return points

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_simple_load_data_no_lag(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        self.assertTrue(all(name in feature.columns for name in ["Date", "Feature1", "Feature2"]))
        self.assertTrue(all(name in log_prices.columns for name in ["Date", "Price"]))
        self.assertEqual(log_prices["Price"].to_list(), fake_price_data)

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_load_data(self, mock):
        feature, log_prices = load_transform_data(None, self.lag_num)
        feature_no_lag, _ = load_transform_data(None, 0)
        true_data = np.array(fake_price_data, dtype=np.float32)
        expected_out = []
        for i in range(len(true_data)-self.lag_num):
            expected_out.append(true_data[i+self.lag_num] - true_data[i])
        self.assertEqual(log_prices["Price"].to_list(), expected_out)
        assert_frame_equal(feature_no_lag.head(len(feature_no_lag)-self.lag_num), feature)
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_simple_load_data_skip(self, mock):
        feature, log_prices = load_transform_data(None, 0, skip=self.skip)
        feature_no_lag, _ = load_transform_data(None, 0)
        true_data = np.array(fake_price_data, dtype=np.float32)
        expected_out = []
        for i in range(len(true_data)-self.skip):
            expected_out.append(true_data[i+self.skip]) 
        self.assertEqual(log_prices["Price"].to_list(), expected_out)
        assert_frame_equal(feature_no_lag.head(len(feature_no_lag)-self.skip), feature)
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_simple_load_data_skip_lag(self, mock):
        feature, log_prices = load_transform_data(None, self.lag_num, skip=self.skip)
        feature_no_lag, _ = load_transform_data(None, 0)
        true_data = np.array(fake_price_data, dtype=np.float32)
        expected_out2 = []
        for i in range(len(true_data)-self.lag_num):
            expected_out2.append(true_data[i+self.lag_num] - true_data[i])
        expected_out = []
        for i in range(len(expected_out2)-self.skip):
            expected_out.append(expected_out2[i+self.skip]) 
        self.assertEqual(log_prices["Price"].to_list(), expected_out)
        assert_frame_equal(feature_no_lag.head(len(feature_no_lag)-self.skip-self.lag_num), feature)
    

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_no_lag_split_partion_no_pad(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        data_points  = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=0,
            is_padding=False,
            convert_date=True,
            offset=-1,
            num_dataset=-1
        )

        all_true_data = [
            (self.the_data.iloc[:3], self.the_data.iloc[3:5]),
            (self.the_data.iloc[5:8], self.the_data.iloc[8:10]),
            (self.the_data.tail(6).head(3), self.the_data.tail(3).head(2)),
        ]

        points = self.getting_points_data(all_true_data)

        self.assertEqual(len(data_points), 5)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-1], points[2])

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_no_lag_split_offset_less(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        data_points  = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=0,
            is_padding=False,
            convert_date=True,
            offset=4,
            num_dataset=-1
        )

        all_true_data = [
            (self.the_data.iloc[:3], self.the_data.iloc[3:5]),
            (self.the_data.iloc[4:7], self.the_data.iloc[7:9]),
            (self.the_data.tail(6).head(3), self.the_data.tail(3).head(2)),
        ]

        points = self.getting_points_data(all_true_data)

        self.assertEqual(len(data_points), 6)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-1], points[2])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_no_lag_split_partion_pad(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        data_points  = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=0,
            is_padding=True,
            convert_date=True,
            offset=-1,
            num_dataset=-1
        )

        all_true_data = [
            (self.the_data.iloc[:3], self.the_data.iloc[3:5]),
            (self.the_data.iloc[5:8], self.the_data.iloc[8:10]),
            (self.the_data.tail(6).head(3), self.the_data.tail(3).head(2)),
            (self.the_data.tail(5).head(3), self.the_data.tail(5).tail(2)),
        ]

        points = self.getting_points_data(all_true_data)

        self.assertEqual(len(data_points), 6)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-2], points[2])
        self.assertEqual(data_points[-1], points[3])

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_no_lag_split_offset_less_pad(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        data_points  = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=0,
            is_padding=True,
            convert_date=True,
            offset=4,
            num_dataset=-1
        )

        all_true_data = [
            (self.the_data.iloc[:3], self.the_data.iloc[3:5]),
            (self.the_data.iloc[4:7], self.the_data.iloc[7:9]),
            (self.the_data.tail(6).head(3), self.the_data.tail(3).head(2)),
            (self.the_data.tail(5).head(3), self.the_data.tail(5).tail(2)),
        ]

        points = self.getting_points_data(all_true_data)

        self.assertEqual(len(data_points), 7)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-2], points[2])
        self.assertEqual(data_points[-1], points[3])

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_no_lag_split_offset_more(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        data_points  = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=0,
            is_padding=False,
            convert_date=True,
            offset=6,
            num_dataset=-1
        )

        all_true_data = [
            (self.the_data.iloc[:3], self.the_data.iloc[3:5]),
            (self.the_data.iloc[6:9], self.the_data.iloc[9:11]),
            (self.the_data.tail(8).head(3), self.the_data.tail(5).head(2)),
        ]

        points = self.getting_points_data(all_true_data)

        self.assertEqual(len(data_points), 4)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-1], points[2])

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_no_lag_split_offset_more_pad(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        data_points  = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=0,
            is_padding=True,
            convert_date=True,
            offset=6,
            num_dataset=-1
        )

        all_true_data = [
            (self.the_data.iloc[:3], self.the_data.iloc[3:5]),
            (self.the_data.iloc[6:9], self.the_data.iloc[9:11]),
            (self.the_data.tail(8).head(3), self.the_data.tail(5).head(2)),
            (self.the_data.tail(5).head(3), self.the_data.tail(5).tail(2)),
        ]

        points = self.getting_points_data(all_true_data)

        self.assertEqual(len(data_points), 5)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-2], points[2])
        self.assertEqual(data_points[-1], points[3])

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_partition_all_no_pad_1(self, mock):
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=False,
            convert_date=True,
            offset=-1,
            num_dataset=-1
        )
        
        # Index of Input and Output 
        all_true_index = [
            (range(3), range(3+6,5+6))
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 1)
        self.assertEqual(data_points[0], points[0])

    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_partition_all_no_pad_2(self, mock):
        self.lag_num = 4
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=False,
            convert_date=True,
            offset=-1,
            num_dataset=-1
        )

        # Index of Input and Output 
        all_true_index = [
            (range(3), range(3+4,5+4)),
            (range(5+4,8+4), range(12+4,14+4))
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 2)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_partition_all_pad_1(self, mock):
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=True,
            convert_date=True,
            offset=-1,
            num_dataset=-1
        )
        
        # Index of Input and Output 
        all_true_index = [
            (range(3), range(3+6,5+6)),
            (range(3+6,6+6), range(12+6,14+6)),
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 2)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_partition_all_pad_2(self, mock):
        self.lag_num = 4
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=True,
            convert_date=True,
            offset=-1,
            num_dataset=-1
        )

        # Index of Input and Output 
        all_true_index = [
            (range(3), range(3+4,5+4)),
            (range(5+4,8+4), range(12+4,14+4)),
            # Don't forget we add the extra-step
            (range(9+4,12+4), range(16+4,18+4)),
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 3)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[2], points[2])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_offset_no_pad_1(self, mock):
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=False,
            convert_date=True,
            offset=2,
            num_dataset=-1
        )

        # Index of Input and Output 
        all_true_index = [
            (range(3), range(9,11)),
            (range(2,5), range(11,13)),
            (range(8,11), range(17,19)),
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 5)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-1], points[-1])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_offset_pad_1(self, mock):
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=True,
            convert_date=True,
            offset=2,
            num_dataset=-1
        )

        # Index of Input and Output 
        all_true_index = [
            (range(3), range(9,11)),
            (range(2,5), range(11,13)),
            (range(8,11), range(17,19)),
            (range(9,12), range(18,20)),
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 6)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[-2], points[2])
        self.assertEqual(data_points[-1], points[3])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_offset_no_pad_2(self, mock):
        self.lag_num = 4
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=False,
            convert_date=True,
            offset=10,
            num_dataset=-1
        )

        # Index of Input and Output 
        all_true_index = [
            (range(3), range(7,9)),
            (range(10,13), range(17,19)),
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 2)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_lag_split_offset_pad_2(self, mock):
        self.lag_num = 4
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=True,
            convert_date=True,
            offset=10,
            num_dataset=-1
        )

        # Index of Input and Output 
        all_true_index = [
            (range(3), range(7,9)),
            (range(10,13), range(17,19)),
            (range(13,16), range(20,22)),
        ]

        points = self.getting_lag_data(log_prices, all_true_index)

        self.assertEqual(len(data_points), 3)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
        self.assertEqual(data_points[2], points[2])
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_convert_date(self, mock):
        self.lag_num = 0
        feature, log_prices = load_transform_data(None, self.lag_num)
        data_points = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=self.lag_num,
            is_padding=True,
            convert_date=False,
            offset=1,
            num_dataset=-1
        )

        self.assertTrue(all(data_points[i].label_inp["Date"].equals(data_points[i].data_inp["Date"]) 
            for i in range(len(data_points))))
    
    @patch("utils.data_preprocessing.load_metal_data", return_value=generate_fake_data())
    def test_num_dataset(self, mock):
        feature, log_prices = load_transform_data(None, 0)
        data_points  = prepare_dataset(
            feature, 
            fake_first_day,
            log_prices,
            self.len_inp,
            self.len_out,
            return_lag=0,
            is_padding=False,
            convert_date=True,
            offset=-1,
            num_dataset=2
        )

        all_true_data = [
            (self.the_data.iloc[:3], self.the_data.iloc[3:5]),
            (self.the_data.iloc[5:8], self.the_data.iloc[8:10]),
            (self.the_data.tail(6).head(3), self.the_data.tail(3).head(2)),
        ]

        points = self.getting_points_data(all_true_data)

        self.assertEqual(len(data_points), 2)
        self.assertEqual(data_points[0], points[0])
        self.assertEqual(data_points[1], points[1])
    