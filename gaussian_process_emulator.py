import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

'''Takes as input a csv file with parameters and the output quantity and its variance'''
random.seed(42)


def form_training_test_set(df_in, test_set_size_in):
    '''
    Split into training and testset
    Inputs:
    df_in, pandas dataframe where each row is a sample
    test_set_size_in, number of sample to remove to a testset

    outputs:
    df_training, subset of df_in chosen as training set
    df_training, subset of df_in chosen as test set
    '''

    total_runs = len(df_in.index)
    test_set_size = test_set_size_in
    if total_runs <= test_set_size_in:
        raise ValueError("All of the data (or more) is assigned to training set, this is not sensible.")

    testset = random.sample(range(total_runs), test_set_size)
    testset.sort()
    df_training = df_in.drop(testset)
    df_test = df_in.iloc[testset]
    return df_training, df_test


df = pd.read_csv("parameters_output.csv")

df_tr, df_te = form_training_test_set(df, 20)

params_tr = df_tr[
    ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym", "T_hos",
     "K", "inf_asym"]]
X_tr = params_tr.to_numpy()  # training parameter combinations
print(X_tr.shape)

output_mean_tr = df_tr["total_deaths_mean"]
y_tr = output_mean_tr.to_numpy()  # training parameter combinations

output_variance_tr = df_tr["total_deaths_variance"]
alpha = output_variance_tr.to_numpy()  # variance at training points, as described in sklearn docs

'''Gaussian process section'''
#Scaling input parameters
scaler = StandardScaler()
scaler.fit(X_tr)

kernel = gp.kernels.ConstantKernel(1.0,[1e-7,1e15]) * gp.kernels.RBF(10.0)
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=alpha)
model.fit(scaler.transform(X_tr), y_tr)
params = model.kernel_.get_params()

'''test'''
params_te = df_te[
    ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym", "T_hos",
     "K", "inf_asym"]]
X_te = params_te.to_numpy()


output_mean_te = df_te["total_deaths_mean"]
y_te = output_mean_te.to_numpy()

y_pred, std = model.predict(scaler.transform(X_te), return_std=True)

model.get_params()


'''plotting validation figure'''
plt.figure()
plt.plot(range(len(y_te)),y_te,'o', color='black', label='True Value')
plt.plot(range(len(y_pred)),y_pred,'o', color='red', label='Prediction')
plt.plot(range(len(y_te)),y_pred - 1.9600 * std,'--',color='black')
plt.plot(range(len(y_te)),y_pred + 1.9600 * std,'--',color='black', label='95% confidence interval')
plt.ylabel('Total hospital deaths over 200 days')
plt.xlabel('Test points')
plt.title('Gaussian Process Emulator Test Set Performance')
plt.legend(loc='lower left')
plt.show()
