import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
import random
'''Takes as input a csv file with parameters and the output quantity and its variance'''
random.seed(42)

df = pd.read_csv("parameters_output.csv")

params= df[["p_inf","p_hcw","c_hcw","d","q","p_s","rrd","lambda","T_lat","juvp_s","T_inf","T_rec","T_sym","T_hos","K","inf_asym"]]
X_tr = params.to_numpy() #training parameter combinations
print(X_tr.shape)

output_mean = df["total_deaths_mean"]
y_tr = output_mean.to_numpy() #training parameter combinations
print(y_tr.shape)

output_variance = df["total_deaths_variance"]
alpha = output_variance.to_numpy() #variance at training points, as described in sklearn docs
print(alpha.shape)

'''Gaussian process section'''
kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(1.0)
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=alpha, normalize_y=True)
model.fit(X_tr, y_tr)
params = model.kernel_.get_params()

'''test'''
#separate into test and training set
#y_pred, std = model.predict(X_te, return_std=True)


