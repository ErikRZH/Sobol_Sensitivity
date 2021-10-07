import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random


def get_training_test_set(df_in, test_set_size_in):
    """
    Split into training and testset
    Inputs:
    df_in, pandas dataframe where each row is a sample
    test_set_size_in, number of sample to remove to a testset

    outputs:
    df_training, subset of df_in chosen as training set
    df_training, subset of df_in chosen as test set
    """

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
    """Returns a model trained on scaled/normalised inputs and scaler object for inputs"""

    #Gaussian process section
    # Scaling input parameters
    gp_scaler = StandardScaler()
    gp_scaler.fit(X_in)

    kernel = gp.kernels.ConstantKernel(1.0, [1e-7, 1e15]) * gp.kernels.RBF(10.0)
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=alpha_in)
    gp_model.fit(gp_scaler.transform(X_in), y_in)

    return gp_model, gp_scaler


def predict_GP_emulator(X_in, model_in, scaler_in, return_std_in: bool = False):
    """
    Args:
        X_in: parameters to predict output for (need not be normalised).
        model_in: model to be evaluated (preferable trained).
        
    Returns:
        y_pred_out: Prediction for parameters given by X_in.
        std_out: standard deviation associated with y_pred_out (if return_std_in = True).
    """

    return model_in.predict(scaler_in.transform(X_in), return_std=return_std_in)


def evaluate_GP_emulator(X_in, y_in, model_in, scaler_in):
    """
    Args:
        X_in: training set parameters (need not be normalised).
        y_in: correct labels for training examples.
        model_in: trained model to be evaluated.
    """
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


def train_and_predict(df_in, params_in, quantity_mean_in, quantity_varaince_in, X_in):
    """

    Takes as input training data and parmeters to be tested and returns the GP emulator predictions
    Args:
        df_in: pandas.dataframe with input parameters and the mean and variance of the output quantity.
        params_in: names of the columns with model parameters in df_in.
        quantity_mean_in: name of the column with the mean of the model output to be analysed.
        quantity_varaiance_in: name of the column with the variance of the model output to be analysed.
        X_in: Numpy array with the parameters (in same order as in params_in) to be tested, and each row a different set of parameters.
    Returns:
        Y_out: the gaussian process estimator predictions for each row in X_in.
    """
    X_tr_temp, y_tr_temp, alpha_tr_temp = form_training_set(df_in,params_in,quantity_mean_in,quantity_varaince_in)
    temp_model, temp_scaler = train_GP_emulator(X_tr_temp, y_tr_temp, alpha_tr_temp)
    y_out = predict_GP_emulator(X_in,temp_model,temp_scaler)
    return y_out


def form_training_set(df_in, params_in, quantity_mean_in, quantity_varaince_in):
    """creates numpy arrays from a pandas dataframe. The numpy arrays can then be used for training
        Returns: NumpyArrays, X_out,Y_out,alpha_out, used for model training
    """
    # Training set
    params_temp = df_in[params_in]
    X_out = params_temp.to_numpy()  # training parameters

    output_mean_temp = df_in[quantity_mean_in]
    y_out = output_mean_temp.to_numpy()  # training outputs

    output_variance_temp = df_in[quantity_varaince_in]
    alpha_out = output_variance_temp.to_numpy()  # variance at training points, as described in sklearn docs
    return X_out, y_out, alpha_out


def form_test_set(df_in, params_in, quantity_mean_in):
    """creates numpy arrays from pandas dataframe. the arrays are the ones used for testing"""
    params_temp = df_in[params_in]  # test set parameters
    X_out = params_temp.to_numpy()

    output_mean_temp = df_in[quantity_mean_in]  # correct test set labels
    y_out = output_mean_temp.to_numpy()
    return X_out, y_out


"""Takes as input a csv file with parameters and the output quantity and its variance"""
def gaussian_process_example():
    """
    This code is an example of how the functions may be used
    """
    df = pd.read_csv("parameters_output.csv")

    parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym",
                  "T_hos",
                  "K", "inf_asym"]
    quantity_mean = "total_deaths_mean"
    quantity_varaince = "total_deaths_variance"

    # Randomly select a few samples to form a training set and a test set
    random.seed(41)
    df_tr, df_te = get_training_test_set(df, 20)

    # Training set
    X_tr, y_tr, alpha = form_training_set(df_tr, parameters, quantity_mean, quantity_varaince)

    # Test set
    X_te, y_te = form_test_set(df_te, parameters, quantity_mean)

    # Training model on training set
    my_model, my_scaler = train_GP_emulator(X_tr, y_tr, alpha)

    # Evaluating results on test set
    evaluate_GP_emulator(X_te, y_te, my_model, my_scaler)
