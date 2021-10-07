from SALib.analyze import sobol
from SALib.sample import saltelli
import gaussian_process_emulator as gpe
import pandas as pd
import numpy as np

def salteli_with_constant_bounds(problem_in,N_in):
    """
    Takes as input a problem dictionary, which may include constants! and an integer N
    provides as output a saletli sampling of the parameters, while ignoring the constants, Returned as a (D,N*(2D+2)) numpy array
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

    return X_sample

'''
As input:
parameters: names of parameters
ranges: ranges of parameters
'''

# ONLY SPECIFY ONCE
parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym",
              "T_hos",
              "K", "inf_asym"]

#a single value means it is be a constant with that value
bounds = [[0, 1], [0, 1], [1, 80], [0, 1], [0, 1], [0, 1], [1],[1e-9, 1e-3], [0.1, 14], [0, 1], [0.1, 21], [1, 28],
          [0.1, 14], [1, 35], [10000], [0, 1]]

quantity_mean = "total_deaths_mean"
quantity_varaince = "total_deaths_variance"
df = pd.read_csv("parameters_output.csv")


#Set up problem dictionary,as needed by SALib
problem_all = {'num_vars': len(parameters),
           'names': parameters,
           'bounds': bounds }

N = 32
# get X to sample
X = salteli_with_constant_bounds(problem_all,N)

# Train and Sample Gaussian Process emulator at points
y = gpe.train_and_predict(df, parameters, quantity_mean, quantity_varaince, X)

print(y)

# Find Sobol
# Si = sobol.analyze(problem, Y)
