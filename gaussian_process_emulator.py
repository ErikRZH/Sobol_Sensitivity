import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random


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


def train_GP_emulator(X_in, y_in, alpha_in):
    '''Returns a model trained on scaled/normalised inputs and scaler object for inputs'''
    '''Gaussian process section'''
    # Scaling input parameters
    gp_scaler = StandardScaler()
    gp_scaler.fit(X_in)

    kernel = gp.kernels.ConstantKernel(1.0, [1e-7, 1e15]) * gp.kernels.RBF(10.0)
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=alpha_in)
    gp_model.fit(gp_scaler.transform(X_in), y_in)

    return gp_model, gp_scaler


def predict_GP_emulator(X_in, model_in, scaler_in, return_std_in: bool = False ):
    '''
    Args:
        X_in: parameters to predict output for (need not be normalised).
        model_in: model to be evaluated (preferable trained).
        
    Returns:
        y_pred_out: Prediction for parameters given by X_in.
        std_out: standard deviation associated with y_pred_out (if return_std_in = True).
    '''

    return model_in.predict(scaler_in.transform(X_in), return_std=return_std_in)


def evaluate_GP_emulator(X_in, y_in, model_in, scaler_in):
    '''
    Args:
        X_in: training set parameters (need not be normalised).
        y_in: correct labels for training examples.
        model_in: trained model to be evaluated.
    '''
    # Predict labels
    y_pred, std = predict_GP_emulator(X_in, model_in, scaler_in, True)

    # Plot validation figure
    plt.figure()
    plt.plot(range(len(y_in)), y_in, 'o', color='black', label='True Value')
    plt.plot(range(len(y_pred)), y_pred, 'o', color='red', label='Prediction')
    plt.plot(range(len(y_te)), y_pred - 1.9600 * std, '--', color='black')
    plt.plot(range(len(y_te)), y_pred + 1.9600 * std, '--', color='black', label='95% confidence interval')
    plt.ylabel('Total hospital deaths over 200 days')
    plt.xlabel('Test points')
    plt.ylim((0))
    plt.title('Gaussian Process Emulator Test Set Performance')
    plt.legend(loc='lower left')
    plt.show()
    return


'''Takes as input a csv file with parameters and the output quantity and its variance'''

df = pd.read_csv("parameters_output.csv")

parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym",
              "T_hos",
              "K", "inf_asym"]
quantity_mean = "total_deaths_mean"
quantity_varaince = "total_deaths_variance"

# Randomly select a few samples to form a training set and a test set
random.seed(41)
df_tr, df_te = form_training_test_set(df, 20)

# Training set
params_tr = df_tr[parameters]
X_tr = params_tr.to_numpy()  # training parameters

output_mean_tr = df_tr[quantity_mean]
y_tr = output_mean_tr.to_numpy()  # training outputs

output_variance_tr = df_tr[quantity_varaince]
alpha = output_variance_tr.to_numpy()  # variance at training points, as described in sklearn docs

# Test set
params_te = df_te[parameters]
X_te = params_te.to_numpy()

output_mean_te = df_te[quantity_mean]
y_te = output_mean_te.to_numpy()

# Training model on training set
my_model, my_scaler = train_GP_emulator(X_tr, y_tr, alpha)

# Evaluating results on test set
evaluate_GP_emulator(X_te, y_te, my_model, my_scaler)
