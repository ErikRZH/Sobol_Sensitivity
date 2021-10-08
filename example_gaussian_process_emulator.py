import gaussian_process_emulator as gpe
import pandas as pd

#Runs the gaussian process emulator example, training and evaluating a model on the included data.

df = pd.read_csv("parameters_output.csv")

parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec",
              "T_sym",
              "T_hos",
              "K", "inf_asym"]
quantity_mean = "total_deaths_mean"
quantity_variance = "total_deaths_variance"
testsetsize = 20

gpe.gaussian_process_example(df,parameters,quantity_mean,quantity_variance,testsetsize)