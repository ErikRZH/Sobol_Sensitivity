from SALib.analyze import sobol
from SALib.sample import saltelli
import gaussian_process_emulator as gpe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def salteli_with_constant_bounds(problem_in, N_in):
    """
    Takes as input a problem dictionary, which may include constants(!), and an integer N
    provides as output a saletli sampling of the parameters, while ignoring the constants, Returned as a (D,N*(2D+2)) numpy array
    Also outputs the problem dictionary with constants removed
    """

    # saletlli_sample_with_constants
    '''removes the constant parameters from the problem dictionary and then generates samples and reinserts constant values'''
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

    '''generates N*(2D+2) samples, where N is argument'''
    X_sample = saltelli.sample(problem_variables, N_in)

    # Reinsert Constant Values
    for i in constant_parameters:
        # Inserts column with the constant value in correct place
        X_sample = np.insert(X_sample, i, np.ones([X_sample.shape[0]]) * problem_in['bounds'][i][0], axis=1)

    return X_sample, problem_variables

def sobol_indicies_from_emulator(df_in, params_in,bounds_in,quantity_mean_in, quantity_variance_in,N_in):
    """

    Computes Sobol indicies from a pandas dataframe using a gaussian emulator
        Args:
            df_in: dataframe with columns being parameters,output mean,output variance. each row is a different sample
            params_in: names of model parameters (can include constants)
            bounds_in: the bounds on the model parameters as [[param1_lower_bound,param1_upper_bound],[...],[param_x_constant],..]. For fixed values of parameters this is the fixed value
            quantity_mean_in: the name of the column containing the mean of the output quantity
            quantity_variance_in: name of the column containing the variance of the output quantity
            N_in: (POWER OF 2!) the number used in parameter sampling (N*(2D+2) samples are generated, D number of non-constant parameters)

        Returns:
            df_out: pandas dataframe with its columns being the 1st and total order sensitivity indicies, and their associated 95% confidence intervals. The rows are the non-constant parameters
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

    #make into pandas dataframe
    total_Si, first_Si, second_Si = Si.to_df()

    df_out = pd.concat([first_Si,total_Si], axis=1)
    return df_out

# ONLY SPECIFY ONCE
def example():
    df = pd.read_csv("parameters_output.csv")

    parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym",
                  "T_hos",
                  "K", "inf_asym"]

    # a single value means it is a constant with that value
    bounds = [[0, 1], [0, 1], [1, 80], [0, 1], [0, 1], [0, 1], [1], [1e-9, 1e-3], [0.1, 14], [0, 1], [0.1, 21], [1, 28],
              [0.1, 14], [1, 35], [10000], [0, 1]]

    quantity_mean = "total_deaths_mean"
    quantity_varaince = "total_deaths_variance"

    N = 2 ** 11  # SELECT A POWER OF 2

    S1_ST_df = sobol_indicies_from_emulator(df,parameters,bounds,quantity_mean,quantity_varaince,N)

    #Conf is the 95% confidence interval

    # Analyse and visualise the results
    labels = S1_ST_df.index.values
    S1_plot = S1_ST_df ["S1"]
    x = np.arange(len(labels))  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x, S1_plot, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('S1')
    ax.set_title('S1 index for parameters')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.bar_label(rects1, padding=3)
    fig.tight_layout()
    plt.show()
