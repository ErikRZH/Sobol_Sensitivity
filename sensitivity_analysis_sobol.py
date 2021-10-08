from SALib.analyze import sobol
from SALib.sample import saltelli
import gaussian_process_emulator as gpe
import pandas as pd
import numpy as np
from typing import Tuple


def salteli_with_constant_bounds(problem_in: dict, N_in: int) -> Tuple[np.ndarray, dict]:
    """

    Takes as input a problem dictionary, which may include constants(!), and an integer N.
    Returns a Saletli sampling of the parameters, ignoring the constants, as a (D,N*(2D+2)) numpy array,
    and the problem dictionary with constants removed

    Arg:
        problem_in: Problem dictionary (keys: 'num_vars', 'names', 'bounds') which may contain constants.
        N_in: N used to determine number of parameter samples in Salteli sampling.  Note: Must be power of 2.

    Returns:
        X_sample: Parameter values to be sampled in Sobol sensitivity analysis.
    """

    # Removes the constant parameters from the problem dictionary, generates samples and reinserts constant values
    constant_parameters = []
    for count, value in enumerate(problem_in['bounds']):
        if len(value) == 1:
            constant_parameters.append(count)
        elif len(value) != 2:
            raise ValueError("parameter named: " + problem_in['names'][
                count] + " has bounds with not 1 entry (constant) or 2 entries (bounds).")

    names_variables = [problem_in['names'][i] for i in range(len(problem_in['names'])) if
                       i not in constant_parameters]
    bounds_variables = [problem_in['bounds'][i] for i in range(len(problem_in['bounds'])) if
                        i not in constant_parameters]

    problem_variables = {'num_vars': len(names_variables),
                         'names': names_variables,
                         'bounds': bounds_variables}

    # Generates N*(2D+2) samples, where N is argument D is number of non-constant varaibles
    X_sample = saltelli.sample(problem_variables, N_in)

    # Reinsert Constant Values
    for i in constant_parameters:
        # Inserts columns with the constant value in correct place
        X_sample = np.insert(X_sample, i, np.ones([X_sample.shape[0]]) * problem_in['bounds'][i][0], axis=1)

    return X_sample, problem_variables


def sobol_indicies_from_emulator(df_in: pd.DataFrame, params_in: list, bounds_in: list, quantity_mean_in: str,
                                 quantity_variance_in: str, N_in: int) -> pd.DataFrame:
    """

    Returns Sobol indices by sampling from a gaussian process emulator trained on model data.

    Args:
        df_in: Dataframe of model data, each row is a different sample.
            Columns are: parameters, output mean, output variance.
        params_in: Names of model parameters (can include constants).
        bounds_in: Bounds of model parameters as [[param1_lower_bound,param1_upper_bound],[...],[param_x_fixed],..].
            For parameters with a fixed value this is the fixed value.
        quantity_mean_in: Name of the column containing the mean of the output quantity.
        quantity_variance_in: Name of the column containing the variance of the output quantity.
        N_in: The number used in parameter sampling, should be a power of 2.
            N*(2D+2) samples are generated where D is the number of non-constant parameters.

    Returns:
        df_out: Dataframe containing 1st and total order sensitivity indicies,
            and associated 95% confidence intervals.
    """
    # Set up problem dictionary,as needed by SALib
    problem_all = {'num_vars': len(params_in),
                   'names': params_in,
                   'bounds': bounds_in}

    # generates N*(2D+2) samples, where N is argument D is number of non-constant variables
    X, problem_reduced = salteli_with_constant_bounds(problem_all, N_in)

    # Train and Sample Gaussian Process emulator at points
    y = gpe.train_and_predict(df_in, params_in, quantity_mean_in, quantity_variance_in, X)

    # Input thereduced problem dictionary, the one with constants removed
    Si = sobol.analyze(problem_reduced, y)

    # make into pandas dataframe
    total_Si, first_Si, second_Si = Si.to_df()

    df_out = pd.concat([first_Si, total_Si], axis=1)
    return df_out
