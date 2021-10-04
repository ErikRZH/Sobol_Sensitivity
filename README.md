# Sobol_Sensitivity
Calculating Sobol Sensitivity Indices (Variance-based sensitivity analysis) given model parameters and outputs.

The procedure is broken up into the following steps:
1) Extracting the output parameter and its variance from a selection of runs.
2) Constructing a gaussian process emulator to sample parameter space more densely.
3) Performing Sobol sensitivity analysis.
