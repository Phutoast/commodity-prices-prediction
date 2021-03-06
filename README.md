# Commodity Prices Prediction
Part of my master project, which includes the following algorithms 

- [x] ARIMA
- [x] Gaussian Process
- [x] Multi-Task Gaussian Process
- [x] Multi-Task Index Gaussian Process (Use index of each task)
- [x] Deep Gaussian Process with Multi-Task Output
- [x] Deep Sigma Point Process with Multi-Task Output
- [x] Sparse Multi-Task Index Gaussian Process
- [x] Sparse Matern Graph Gaussian Process
- [x] Deep Graph Kernel 
- [x] Deep Graph Kernel + Deep Graph Infomax Pretraining
- [ ] Cluster Multi-Task GP (Pyro + Gpytorch)
- [x] Non-Linear Deep Multi-Task GP
- [x] Non-Linear Deep Sigma Point Process
- [ ] Cluster Non-Linear Deep Multi-Task GP
- [ ] Cluster Non-Linear Deep Sigma Point Process
- [ ] Learning Graph GP
- [x] Graph Propagation Deep GP
- [x] Interaction Net Deep GP
- [x] DSPP Graph Propagation GP
- [x] Interaction Net DSPP
- [ ] Non-Linear Deep Multi-Task GP Multi-Output
- [ ] Non-Linear Deep Sigma Point Process Multi-Output


See `main.py` for examples. Running a Test for Data-Splitting Algorithm. The data should be stored in `data/{metal_name}`.


In order to run the experiments, we assume to have a `raw_data` folder that contains folders that named after the commodities, which have `{commodity name}_feature.csv` and `{commodity name}_raw_prices.csv` (this should be raw prices *not* log of it) storted within. To create a preprocessed data that is saved in folder `data`, we run `save_date_common("raw_data", "data")` from `utils.data_preprocessing`.

We can run the test by:
```sh
python -m pytest
```

For example, we have:
![alt text](img/walk_forward.png)

One may be interested in training the GP within google colabs, we have provided a simple way to zip the necessary files/folder

```sh
sh upload/zip_folders.sh
```
where we can upload to the colabs, extract the file and then perform the training.