import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import random


def split_training_test_set(df_in: pd.DataFrame, test_set_size_in: int, seed: int = 42) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """

    Splits a Dataframe into two, by randomly sampling rows, so one can be used for training and the other for testing.

    Args:
        df_in: Dataframe of model data, each row is a different sample.
        Columns are, parameters, output mean, output variance.
        test_set_size_in: Number of samples to move to a test set.
        seed: Random seed for drawing the test set.  Default: 42.

    Returns:
        df_training: Subset of df_in chosen as training set.
        df_test: Subset of df_in chosen as test set.
    """

    total_runs = len(df_in.index)
    test_set_size = test_set_size_in
    if total_runs <= test_set_size_in:
        raise ValueError("All of the data (or more) is assigned to training set, this is not sensible.")

    random.seed(seed)
    testset = random.sample(range(total_runs), test_set_size)
    testset.sort()
    df_training = df_in.drop(testset)
    df_test = df_in.iloc[testset]
    return df_training, df_test


def train_GP_emulator(X_in: np.ndarray, y_in: np.ndarray, alpha_in: np.ndarray) -> Tuple[
    gp.GaussianProcessRegressor, StandardScaler]:
    """

    Returns a model trained on scaled inputs, also returns the associated Scaler object.

    Args:
        X_in: Model parameter values, need not be normalised.
        y_in: Model output values, need not be normalised (they would never need to anyway).
        alpha_in: The variance of the model outputs. From SKLearn Docs:
            https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
            " It can be interpreted as the variance of additional Gaussian measurement noise on the training observations".

    Returns:
        gp_model: Trained Gaussian model object.
        gp_scaler: Scaler object for input data.
    """

    # Gaussian process section
    # Scaling input parameters
    gp_scaler = StandardScaler()
    gp_scaler.fit(X_in)

    kernel = gp.kernels.ConstantKernel(1.0, [1e-7, 1e15]) * gp.kernels.RBF(10.0)
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=alpha_in)
    gp_model.fit(gp_scaler.transform(X_in), y_in)

    return gp_model, gp_scaler


def predict_GP_emulator(X_in: np.ndarray, model_in: gp.GaussianProcessRegressor, scaler_in: StandardScaler,
                        return_std_in: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """

    Makes predictions given parameter values, a model and a Scaler.

    Args:
        X_in: Parameters to predict output for, need not be normalised.
        model_in: Model to be evaluated, preferably a trained model.
        scaler_in: The Scaler associated with the trained model.
        return_std_in: Whether or not to return the standard deviation of the predictions
        
    Returns:
        y_pred: Prediction of output for parameters given by X_in.
        std_out: standard deviation associated with y_pred_out (if return_std_in = True).
    """

    return model_in.predict(scaler_in.transform(X_in), return_std=return_std_in)


def evaluate_GP_emulator(X_in: np.ndarray, y_in: np.ndarray, model_in: gp.GaussianProcessRegressor,
                         scaler_in: StandardScaler):
    """

    Evaluate the Gaussian process emulator on a test set and plot the results.
    Do not use the set you used for training...

    Args:
        X_in: Test set parameters, need not be normalised.
        y_in: Correct output for training examples.
        model_in: Model to be evaluated.
        scaler_in: The Scaler associated with the model.
    """
    # Predict labels
    y_pred, std = predict_GP_emulator(X_in, model_in, scaler_in, True)

    # Plot validation figure
    plt.figure()
    plt.plot(range(len(y_in)), y_in, 'o', color='black', label='True Value')
    plt.plot(range(len(y_pred)), y_pred, 'o', color='red', label='Prediction')
    plt.plot(range(len(y_in)), y_pred - 1.9600 * std, '--', color='black')
    plt.plot(range(len(y_in)), y_pred + 1.9600 * std, '--', color='black', label='95% confidence interval')
    plt.ylabel('Total Hospital Deaths over 200 days')
    plt.xlabel('Test points')
    plt.ylim(0)
    plt.title('Gaussian Process Emulator Test Set Performance')
    plt.legend(loc='lower left')
    plt.show()
    return


def train_and_predict(df_in: pd.DataFrame, params_in: list, quantity_mean_in: str, quantity_variance_in: str,
                      X_in: np.ndarray) -> np.ndarray:
    """

    Takes training data and parameter values to be evaluated and returns the GP emulator predictions at those values.

    Args:
        df_in: Dataframe with input parameters and the mean and variance of the output quantity.
        params_in: Names of the columns with model parameters in df_in.
        quantity_mean_in: name of the column with the mean of the model output to be analysed.
        quantity_variance_in: name of the column with the variance of the model output to be analysed.
        X_in: Numpy array with the parameters to be tested, each row being a different set of parameters.

    Returns:
        y_out: Gaussian process estimator predictions for each set of parameters in X_in.
    """
    X_tr_temp, y_tr_temp, alpha_tr_temp = form_training_set(df_in, params_in, quantity_mean_in, quantity_variance_in)
    temp_model, temp_scaler = train_GP_emulator(X_tr_temp, y_tr_temp, alpha_tr_temp)
    y_out = predict_GP_emulator(X_in, temp_model, temp_scaler)
    return y_out


def form_training_set(df_in: pd.DataFrame, params_in: list, quantity_mean_in: str, quantity_variance_in: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forms numpy arrays from a dataframe, the numpy arrays can then be used for training a model.

    Args:
        df_in: Dataframe with input parameters and the mean and variance of the output quantity.
        params_in: Names of the columns with model parameters in df_in.
        quantity_mean_in: Name of the column with the mean of the model output to be analysed.
        quantity_variance_in: Name of the column with the variance of the model output to be analysed.

    Returns:
        X_out: Numpy array with training set parameters, used for training a model.
        y_out: Numpy array with training set outputs, used for training a model.
        alpha_out: Numpy array with the variance of the training set outputs, can be used for training a model.
    """

    # Training set
    params_temp = df_in[params_in]
    X_out = params_temp.to_numpy()  # training parameters

    output_mean_temp = df_in[quantity_mean_in]
    y_out = output_mean_temp.to_numpy()  # training outputs

    output_variance_temp = df_in[quantity_variance_in]
    alpha_out = output_variance_temp.to_numpy()  # variance at training points, as described in sklearn docs
    return X_out, y_out, alpha_out


def form_test_set(df_in: pd.DataFrame, params_in: list, quantity_mean_in: str) -> Tuple[np.ndarray, np.ndarray]:
    """

    Forms numpy arrays from a dataframe, the numpy arrays can be used for testing and evaluating a model.

    Args:
        df_in: Dataframe with input parameters and the mean and variance of the output quantity.
        params_in: Names of the columns with model parameters in df_in.
        quantity_mean_in: Name of the column with the mean of the model output.

    Returns:
        X_out: Numpy array with test set parameters.
        y_out: Numpy array with test set outputs.

    """
    params_temp = df_in[params_in]  # test set parameters
    X_out = params_temp.to_numpy()

    output_mean_temp = df_in[quantity_mean_in]  # correct test set labels
    y_out = output_mean_temp.to_numpy()
    return X_out, y_out


def gaussian_process_example(df_in: pd.DataFrame, params_in: list, quantity_mean_in: str, quantity_variance_in: str, testsetsize_in: int):
    """

    This is an example of how the gaussian_process_emulator functions may be used.
    It returns a plot for evaluating model performance

    Args:
        df_in: Dataframe with input parameters and the mean and variance of the output quantity.
        params_in: Names of the columns with model parameters in df_in.
        quantity_mean_in: Name of the column with the mean of the model output to be analysed.
        quantity_variance_in: Name of the column with the variance of the model output to be analysed.
        testsetsize_in: Size of the test set to draw from the data.
    """


    # Randomly select a few samples to form a training set and a test set
    df_tr, df_te = split_training_test_set(df_in, testsetsize_in)

    # Training set
    X_tr, y_tr, alpha = form_training_set(df_tr, params_in, quantity_mean_in, quantity_variance_in)

    # Test set
    X_te, y_te = form_test_set(df_te, params_in, quantity_mean_in)

    # Train model on training set
    my_model, my_scaler = train_GP_emulator(X_tr, y_tr, alpha)

    # Evaluate results on test set
    evaluate_GP_emulator(X_te, y_te, my_model, my_scaler)
