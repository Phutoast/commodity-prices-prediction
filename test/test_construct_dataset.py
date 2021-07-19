import numpy as np
import pandas as pd
from utils.data_preprocessing import load_dataset_from_desc
from utils.data_structure import DatasetTaskDesc

import unittest
from unittest.mock import patch
from test.test_utils import generate_fake_data, fake_price_data, fake_first_day, get_loc_non_nan
from pandas.testing import assert_frame_equal, assert_series_equal

import pytest

class ConstructMergeDataset(unittest.TestCase): 
    """
    The Return Lag and the Skip doesn't depends on the construction 
        So we didn't test it here, see test_load_data for the test.
    """

    def setUp(self):
        pass


    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_no_date(self): 
        with self.assertRaises(ValueError) as context:
            simple_desc = DatasetTaskDesc(
                inp_metal_list=["metal1"],
                use_feature=["metal1.Feature1"],
                use_feat_tran_lag=[None],
                out_feature="metal1.Price",
                out_feat_tran_lag=(0, 0, "log"),
            )
        
        self.assertTrue(
            "Date has to be included in use_feature" in str(context.exception)
        )
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_data_leak(self): 
        with pytest.warns(UserWarning, match="Duplication between the output column and Feature"):
            simple_desc = DatasetTaskDesc(
                inp_metal_list=["metal1"],
                use_feature=["Date", "metal1.Feature1"],
                use_feat_tran_lag=[None, None],
                out_feature="metal1.Feature1",
                out_feat_tran_lag=(0, 0, "log"),
            )
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_feature_trans_inp_1(self): 
        with self.assertRaises(ValueError) as context:
            simple_desc = DatasetTaskDesc(
                inp_metal_list=["metal1"],
                use_feature=["Date", "metal1.Feature1-1"],
                use_feat_tran_lag=[None],
                out_feature="metal1.Feature2",
                out_feat_tran_lag=(0, 0, "log"),
            )
        
        self.assertTrue(
            "If defining use_feat_tran_lag to be a list, the length of it should be the same as use_feature" in str(context.exception)
        )
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None],
            out_feature="metal1.Price",
            out_feat_tran_lag=None,
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_transform_date(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[(0,0,"log")],
            out_feature="metal1.Price",
            out_feat_tran_lag=None,
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_no_change_trans_out(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None],
            out_feature="metal1.Price",
            out_feat_tran_lag=(0, 0, "sin"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_diff_out_1(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        print(target[["Output"]])
        print(real_out)

        assert_frame_equal(target[["Output"]], real_out)
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_diff_out_transform_out(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 0, "sin"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        assert_frame_equal(target[["Output"]], np.sin(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_mulit_feat_1(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, None],
            out_feature="metal1.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (real_data["FeatureFamily.Feature1"], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_mulit_feat_diff_out(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, None],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (real_data["FeatureFamily.Feature1"], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_mulit_feat_diff_out_no_transform(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (real_data["FeatureFamily.Feature1"], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_mulit_feat_diff_out_feat_trans(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (0, 0, "sin")],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (np.sin(real_data["FeatureFamily.Feature1"]), feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]
        assert_frame_equal(target[["Output"]], real_out)
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_mulit_feat_diff_out_price_trans(self):
        pred_feature_out = ["Date", "metal1.Price"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (np.log(real_data["Price"]), feature["metal1.Price"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(5, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (real_data["FeatureFamily.Feature1"][:exp_len_data], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(fake_data[i+5] - fake_data[i])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag_transform(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(5, 0, "sin"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (real_data["FeatureFamily.Feature1"][:exp_len_data], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.sin(fake_data[i+5]) - np.sin(fake_data[i]))
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_skip(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 5, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (real_data["FeatureFamily.Feature1"][:exp_len_data], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(fake_data[i+5])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_skip_transform(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 5, "sin"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (real_data["FeatureFamily.Feature1"][:exp_len_data], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.sin(fake_data[i+5]))
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_skip_lag(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(3, 2, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (real_data["FeatureFamily.Feature1"][:exp_len_data], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(fake_data[i+3] - fake_data[i])
        expected_out_2 = []
        for i in range(len(expected_out)-2):
            expected_out_2.append(expected_out[i+2])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out_2
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_skip_lag_transform(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(3, 2, "sin"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (real_data["FeatureFamily.Feature1"][:exp_len_data], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(np.sin(fake_data[i+3]) - np.sin(fake_data[i]))
        expected_out_2 = []
        for i in range(len(expected_out)-2):
            expected_out_2.append(expected_out[i+2])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out_2
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag_compose(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (2, 0, "id")],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(3, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-3

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data = real_data["FeatureFamily.Feature1"]
        expected_out = []
        for i in range(len(fake_data)-2):
            expected_out.append(fake_data[i+2] - fake_data[i])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out[:-1]), feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(fake_data[i+3] - fake_data[i])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag_compose_small_out(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (3, 0, "id")],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(2, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-3

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data = real_data["FeatureFamily.Feature1"]
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(fake_data[i+3] - fake_data[i])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out), feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-2):
            expected_out.append(fake_data[i+2] - fake_data[i])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out[:-1]
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag_compose_transform_feat_only(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (2, 0, "sin")],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(3, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-3

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data = real_data["FeatureFamily.Feature1"]
        expected_out = []
        for i in range(len(fake_data)-2):
            expected_out.append(np.sin(fake_data[i+2]) - np.sin(fake_data[i]))

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out[:-1]), feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(fake_data[i+3] - fake_data[i])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag_skip_compose(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (0, 2, "id")],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(3, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-3

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data = real_data["FeatureFamily.Feature1"]
        expected_out = []
        for i in range(len(fake_data)-2):
            expected_out.append(fake_data[i+2])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out[:-1]), feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(fake_data[i+3] - fake_data[i])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag_skip_compose_skip_out(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (3, 0, "id")],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 2, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-3

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data = real_data["FeatureFamily.Feature1"]
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(fake_data[i+3] - fake_data[i])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out), feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-2):
            expected_out.append(fake_data[i+2])
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out[:-1]
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_out_lag_skip_compose_transform_skip_out(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (3, 0, "id")],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 2, "sin"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-3

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data = real_data["FeatureFamily.Feature1"]
        expected_out = []
        for i in range(len(fake_data)-3):
            expected_out.append(fake_data[i+3] - fake_data[i])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out), feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))

        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-2):
            expected_out.append(np.sin(fake_data[i+2]))
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out[:-1]
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None],
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source_out_non_price(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None],
            out_feature="metal2.Feature2",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source_feat(self):
        pred_feature_out = ["Date", "metal1.Feature2", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (real_data["FeatureFamily.Feature2"], feature["metal1.Feature2"]),
            (real_data_2["FeatureFamily.Feature3"], feature["metal2.Feature3"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source_feat_transform(self):
        pred_feature_out = ["Date", "metal1.Feature2", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (0, 0, "sin"), None],
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (np.sin(real_data["FeatureFamily.Feature2"]), feature["metal1.Feature2"]),
            (real_data_2["FeatureFamily.Feature3"], feature["metal2.Feature3"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source_feat_lag_2_feats(self):
        pred_feature_out = ["Date", "metal1.Feature2", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (3, 0, "id"),(5, 0, "id")],
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data_1 = real_data["FeatureFamily.Feature2"]
        expected_out_1 = []
        for i in range(len(fake_data_1)-3):
            expected_out_1.append(fake_data_1[i+3] - fake_data_1[i])
        
        fake_data_2 = real_data_2["FeatureFamily.Feature3"]
        expected_out_2 = []
        for i in range(len(fake_data_2)-5):
            expected_out_2.append(fake_data_2[i+5] - fake_data_2[i])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out_1)[:-2], feature["metal1.Feature2"]),
            (pd.Series(expected_out_2), feature["metal2.Feature3"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["Price"]][:exp_len_data]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source_feat_skip_2_feats(self):
        pred_feature_out = ["Date", "metal1.Feature2", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (0, 3, "id"),(0, 5, "id")],
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data_1 = real_data["FeatureFamily.Feature2"]
        expected_out_1 = []
        for i in range(len(fake_data_1)-3):
            expected_out_1.append(fake_data_1[i+3])
        
        fake_data_2 = real_data_2["FeatureFamily.Feature3"]
        expected_out_2 = []
        for i in range(len(fake_data_2)-5):
            expected_out_2.append(fake_data_2[i+5])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out_1)[:-2], feature["metal1.Feature2"]),
            (pd.Series(expected_out_2), feature["metal2.Feature3"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["Price"]][:exp_len_data]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source_feat_lag_skip_feats(self):
        pred_feature_out = ["Date", "metal1.Feature2", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (0, 3, "id"),(5, 0, "id")],
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data_1 = real_data["FeatureFamily.Feature2"]
        expected_out_1 = []
        for i in range(len(fake_data_1)-3):
            expected_out_1.append(fake_data_1[i+3])
        
        fake_data_2 = real_data_2["FeatureFamily.Feature3"]
        expected_out_2 = []
        for i in range(len(fake_data_2)-5):
            expected_out_2.append(fake_data_2[i+5] - fake_data_2[i])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out_1)[:-2], feature["metal1.Feature2"]),
            (pd.Series(expected_out_2), feature["metal2.Feature3"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["Price"]][:exp_len_data]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_mult_source_feat_lag_skip_everything(self):
        pred_feature_out = ["Date", "metal1.Feature2", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (0, 3, "id"),(5, 0, "id")],
            out_feature="metal2.Price",
            out_feat_tran_lag=(6, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-6

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        
        fake_data_1 = real_data["FeatureFamily.Feature2"]
        expected_out_1 = []
        for i in range(len(fake_data_1)-3):
            expected_out_1.append(fake_data_1[i+3])
        
        fake_data_2 = real_data_2["FeatureFamily.Feature3"]
        expected_out_2 = []
        for i in range(len(fake_data_2)-5):
            expected_out_2.append(fake_data_2[i+5] - fake_data_2[i])

        real_feature = [
            (real_data["Date"][:exp_len_data], feature["Date"]),
            (pd.Series(expected_out_1)[:-3], feature["metal1.Feature2"]),
            (pd.Series(expected_out_2)[:-1], feature["metal2.Feature3"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_data = generate_fake_data("metal2")
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        fake_data = real_out["Output"].to_list()
        expected_out = []
        for i in range(len(fake_data)-6):
            expected_out.append(np.log(fake_data[i+6]) - np.log(fake_data[i]))
        
        real_out = real_out.iloc[:exp_len_data, :]
        real_out["Output"] = expected_out
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_weird=True)
    )
    def test_load_data_price_mult_source_weird_out(self):
        pred_feature_out = ["Date"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None],
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2", is_weird=True)

        self.assertEqual(feature.columns, pred_feature_out)
        real_feature = [
            (real_data_2["Date"], feature["Date"])
        ] 
        self.assertTrue(all(
            real.to_list() == our_out.to_list() for real, our_out in real_feature
        ))
        
        real_out = real_data_2[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        np.testing.assert_allclose(
            target["Output"].to_numpy(), 
            np.log(real_out)["Output"].to_numpy()
        )

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_weird=True)
    )
    def test_load_data_price_mult_source_weird_mult_feature(self):
        pred_feature_out = ["Date", "metal1.Feature2", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal2.Price",
            out_feat_tran_lag=(0, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        real_data_2 = generate_fake_data("metal2", is_weird=True)
        real_data = real_data.loc[real_data_2.index]

        self.assertEqual(list(feature.columns), pred_feature_out)

        real_feature = [
            (real_data_2["Date"], feature["Date"]),
            (real_data["FeatureFamily.Feature2"], feature["metal1.Feature2"]),
            (real_data_2["FeatureFamily.Feature3"], feature["metal2.Feature3"]),
        ] 
        self.assertTrue(all(
            real.to_list() == our_out.to_list() for real, our_out in real_feature
        ))
        
        real_out = real_data_2[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        np.testing.assert_allclose(
            target["Output"].to_numpy(), 
            np.log(real_out["Output"]).to_numpy()
        )
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_nan=True)
    )
    def test_load_data_drop_nan_normal(self):
        # Only remove the data when feature are not avaliable. 
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, None],
            out_feature="metal1.Price",
            out_feat_tran_lag=(0, 0, "id"),
            is_drop_nan=True
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        loc_non_nan = get_loc_non_nan(1, [1])

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"].loc[loc_non_nan], feature["Date"]),
            (real_data["FeatureFamily.Feature1"].loc[loc_non_nan], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]].loc[loc_non_nan]
        real_out.columns = ["Output"]
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_nan=True)
    )
    def test_load_data_drop_nan_normal_block_both_feature(self):
        # Only remove the data when feature are not avaliable. 
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, None],
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(0, 0, "id"),
            is_drop_nan=True
        )
        with pytest.warns(UserWarning, match="There is a NaN in the output, we can still drop it but the time series may be irregular."):
            feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        loc_non_nan = get_loc_non_nan(1, [1, 2])

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"].loc[loc_non_nan], feature["Date"]),
            (real_data["FeatureFamily.Feature1"].loc[loc_non_nan], feature["metal1.Feature1"])
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["FeatureFamily.Feature2"]].loc[loc_non_nan]
        real_out.columns = ["Output"]
        assert_frame_equal(target[["Output"]], real_out)
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_nan=True)
    )
    def test_load_data_drop_nan_retain_trans_lag(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Price",
            out_feat_tran_lag=(5, 0, "id"),
            is_drop_nan=True
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        loc_non_nan = get_loc_non_nan(1, [1])
        exp_len_data = total_len_data-5
        loc_non_nan = list(filter(lambda x: x < exp_len_data, loc_non_nan))

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"].iloc[loc_non_nan], feature["Date"]),
            (real_data["FeatureFamily.Feature1"].iloc[loc_non_nan], feature["metal1.Feature1"])
        ] 

        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()

        # We transform the data FIRST before we do anything more.
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.log(fake_data[i+5]) - np.log(fake_data[i]))
        
        expected_out = np.array(expected_out)[loc_non_nan]
        
        real_out = real_out.iloc[loc_non_nan]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_nan=True)
    )
    def test_load_data_drop_nan_retain_trans_skip(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Price",
            out_feat_tran_lag=(0, 5, "id"),
            is_drop_nan=True
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        loc_non_nan = get_loc_non_nan(1, [1])
        exp_len_data = total_len_data-5
        loc_non_nan = list(filter(lambda x: x < exp_len_data, loc_non_nan))

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"].iloc[loc_non_nan], feature["Date"]),
            (real_data["FeatureFamily.Feature1"].iloc[loc_non_nan], feature["metal1.Feature1"])
        ] 

        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()

        # We transform the data FIRST before we do anything more.
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.log(fake_data[i+5]))
        
        expected_out = np.array(expected_out)[loc_non_nan]
        
        real_out = real_out.iloc[loc_non_nan]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_nan=True)
    )
    def test_load_data_drop_nan_retain_trans_skip(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Price",
            out_feat_tran_lag=(0, 5, "id"),
            is_drop_nan=True
        )
        feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        loc_non_nan = get_loc_non_nan(1, [1])
        exp_len_data = total_len_data-5
        loc_non_nan = list(filter(lambda x: x < exp_len_data, loc_non_nan))

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"].iloc[loc_non_nan], feature["Date"]),
            (real_data["FeatureFamily.Feature1"].iloc[loc_non_nan], feature["metal1.Feature1"])
        ] 

        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_list()

        # We transform the data FIRST before we do anything more.
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.log(fake_data[i+5]))
        
        expected_out = np.array(expected_out)[loc_non_nan]
        
        real_out = real_out.iloc[loc_non_nan]
        real_out["Output"] = expected_out
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_nan=True)
    )
    def test_load_data_drop_nan_lag_output_Nan(self):
        pred_feature_out = ["Date", "metal1.Feature1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(5, 0, "id"),
            is_drop_nan=True
        )
        with pytest.warns(UserWarning, match="There is a NaN in the output, we can still drop it but the time series may be irregular."):
            feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5
        
        loc_non_nan_1 = get_loc_non_nan(1, [1])
        loc_non_nan_1 = list(filter(lambda x: x < exp_len_data, loc_non_nan_1))

        loc_non_nan_2 = get_loc_non_nan(1, [2])
        loc_nan_2 = [i for i in range(total_len_data) if i not in loc_non_nan_2]
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_numpy()
        fake_data[loc_nan_2] = np.nan

        # We transform the data FIRST before we do anything more.
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(fake_data[i+5]-fake_data[i])
        
        loc_not_nan_2 = np.nonzero(~np.isnan(expected_out))[0]
        all_loc_not_nan = sorted(
            list(set(loc_not_nan_2).intersection(set(loc_non_nan_1)))
        )

        # -------------------------------------------

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"].iloc[all_loc_not_nan], feature["Date"]),
            (real_data["FeatureFamily.Feature1"].iloc[all_loc_not_nan], feature["metal1.Feature1"])
        ] 

        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
         
        real_out = real_out.iloc[all_loc_not_nan]
        real_out["Output"] = np.array(expected_out)[all_loc_not_nan]
        
        assert_frame_equal(target[["Output"]], real_out)

    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=lambda x, y: generate_fake_data(x, is_nan=True)
    )
    def test_load_data_drop_nan_lag_other_out_nan_out(self):
        pred_feature_out = ["Date", "metal1.Feature1", "metal2.Feature3"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1", "metal2"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Feature2",
            out_feat_tran_lag=(5, 0, "id"),
            is_drop_nan=True
        )

        with pytest.warns(UserWarning, match="There is a NaN in the output, we can still drop it but the time series may be irregular."):
            feature, target = load_dataset_from_desc(simple_desc)

        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5
        
        loc_non_nan_1 = get_loc_non_nan(1, [1, 3])
        loc_non_nan_1 = list(filter(lambda x: x < exp_len_data, loc_non_nan_1))

        loc_non_nan_2 = get_loc_non_nan(1, [2])
        loc_nan_2 = [i for i in range(total_len_data) if i not in loc_non_nan_2]
        
        real_out = real_data[["FeatureFamily.Feature2"]]
        real_out.columns = ["Output"]

        fake_data = real_out["Output"].to_numpy()
        fake_data[loc_nan_2] = np.nan

        # We transform the data FIRST before we do anything more.
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(fake_data[i+5]-fake_data[i])
        
        loc_not_nan_2 = np.nonzero(~np.isnan(expected_out))[0]
        all_loc_not_nan = sorted(
            list(set(loc_not_nan_2).intersection(set(loc_non_nan_1)))
        )

        # -------------------------------------------

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"].iloc[all_loc_not_nan], feature["Date"]),
            (real_data["FeatureFamily.Feature1"].iloc[all_loc_not_nan], feature["metal1.Feature1"])
        ] 

        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
         
        real_out = real_out.iloc[all_loc_not_nan]
        real_out["Output"] = np.array(expected_out)[all_loc_not_nan]
        
        assert_frame_equal(target[["Output"]], real_out) 
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_same_feature_name_no_lag(self):
        pred_feature_out = ["Date", "metal1.Price"]
        with pytest.warns(UserWarning, match="Duplication between the output column and Feature"):
            simple_desc = DatasetTaskDesc(
                inp_metal_list=["metal1"],
                use_feature=pred_feature_out,
                use_feat_tran_lag=None,
                out_feature="metal1.Price",
                out_feat_tran_lag=None,
            )
        feature, target = load_dataset_from_desc(simple_desc)
        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), ["Date", "metal1.Price"])
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (np.log(real_data["Price"]), feature["metal1.Price"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_same_feature_name_lag(self):
        pred_feature_out = ["Date", "metal1.Price"]
        with pytest.warns(UserWarning, match="Duplication between the output column and Feature"):
            simple_desc = DatasetTaskDesc(
                inp_metal_list=["metal1"],
                use_feature=pred_feature_out,
                use_feat_tran_lag=[None, (5, 0, "id")],
                out_feature="metal1.Price",
                out_feat_tran_lag=None,
            )
        feature, target = load_dataset_from_desc(simple_desc)
        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), ["Date", "metal1.Price"])

        fake_data = real_data["Price"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.log(fake_data[i+5]) - np.log(fake_data[i]))

        real_feature = [
            (real_data["Date"][:exp_len_data].to_list(), feature["Date"].to_list()),
            (expected_out, feature["metal1.Price"].to_list()),
        ] 
        self.assertTrue(all(
            real == our_out for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]][:exp_len_data]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_same_feature_version_marked(self):
        pred_feature_out = ["Date", "metal1.Price-1"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Price",
            out_feat_tran_lag=None,
        )
        feature, target = load_dataset_from_desc(simple_desc)
        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), ["Date", "metal1.Price-1"])
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (np.log(real_data["Price"]), feature["metal1.Price-1"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_same_feature_multiple_name_same_val(self):
        pred_feature_out = ["Date", "metal1.Price-1", "metal1.Price-2"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=None,
            out_feature="metal1.Price",
            out_feat_tran_lag=None,
        )
        feature, target = load_dataset_from_desc(simple_desc)
        real_data = generate_fake_data("metal1")

        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"], feature["Date"]),
            (np.log(real_data["Price"]), feature["metal1.Price-1"]),
            (np.log(real_data["Price"]), feature["metal1.Price-2"]),
        ] 
        self.assertTrue(all(
            real.equals(our_out) for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_same_feature_lag_one_version(self):
        pred_feature_out = ["Date", "metal1.Price-1", "metal1.Price-2"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (5, 0, "id"), None],
            out_feature="metal1.Price",
            out_feat_tran_lag=None,
        )
        feature, target = load_dataset_from_desc(simple_desc)
        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        fake_data = real_data["Price"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.log(fake_data[i+5]) - np.log(fake_data[i]))

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data].to_list(), feature["Date"].to_list()),
            (expected_out, feature["metal1.Price-1"].to_list()),
            (np.log(real_data["Price"])[:exp_len_data].to_list(), feature["metal1.Price-2"].to_list()),
        ] 
        self.assertTrue(all(
            real == our_out for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]][:exp_len_data]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_same_feature_more_than_1_lag(self):
        pred_feature_out = ["Date", "metal1.Price-1", "metal1.Price-2"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (5, 0, "id"), (3, 0, "id")],
            out_feature="metal1.Price",
            out_feat_tran_lag=None,
        )
        feature, target = load_dataset_from_desc(simple_desc)
        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        fake_data = real_data["Price"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.log(fake_data[i+5]) - np.log(fake_data[i]))
        
        expected_out_2 = []
        for i in range(len(fake_data)-3):
            expected_out_2.append(np.log(fake_data[i+3]) - np.log(fake_data[i]))

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data].to_list(), feature["Date"].to_list()),
            (expected_out, feature["metal1.Price-1"].to_list()),
            (expected_out_2[:exp_len_data], feature["metal1.Price-2"].to_list()),
        ] 

        self.assertTrue(all(
            real == our_out for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]][:exp_len_data]
        real_out.columns = ["Output"]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], np.log(real_out))
    
    @patch(
        "utils.data_preprocessing.load_metal_data", 
        new=generate_fake_data
    )
    def test_load_data_price_same_feature_all_lag_output_too(self):
        pred_feature_out = ["Date", "metal1.Price-1", "metal1.Price-2"]
        simple_desc = DatasetTaskDesc(
            inp_metal_list=["metal1"],
            use_feature=pred_feature_out,
            use_feat_tran_lag=[None, (5, 0, "id"), (3, 0, "id")],
            out_feature="metal1.Price",
            out_feat_tran_lag=(1, 0, "id"),
        )
        feature, target = load_dataset_from_desc(simple_desc)
        real_data = generate_fake_data("metal1")
        total_len_data = len(real_data)
        exp_len_data = total_len_data-5

        fake_data = real_data["Price"].to_list()
        expected_out = []
        for i in range(len(fake_data)-5):
            expected_out.append(np.log(fake_data[i+5]) - np.log(fake_data[i]))
        
        expected_out_2 = []
        for i in range(len(fake_data)-3):
            expected_out_2.append(np.log(fake_data[i+3]) - np.log(fake_data[i]))
        
        expected_out_3 = []
        for i in range(len(fake_data)-1):
            expected_out_3.append(np.log(fake_data[i+1]) - np.log(fake_data[i]))

        self.assertEqual(len(feature), len(target), exp_len_data)
        self.assertEqual(list(feature.columns), pred_feature_out)
        real_feature = [
            (real_data["Date"][:exp_len_data].to_list(), feature["Date"].to_list()),
            (expected_out, feature["metal1.Price-1"].to_list()),
            (expected_out_2[:exp_len_data], feature["metal1.Price-2"].to_list()),
        ] 

        self.assertTrue(all(
            real == our_out for real, our_out in real_feature
        ))
        
        real_out = real_data[["Price"]][:exp_len_data]
        real_out.columns = ["Output"]
        real_out["Output"] = expected_out_3[:exp_len_data]
        
        # Note That we setout that all price should at least be log
        assert_frame_equal(target[["Output"]], real_out)