'this class parses model runs and returns a CSV with the parameters and mean and variance of the output to be analysed'
import csv
import pandas as pd
import numpy as np


def data_extraction_sum(location, iter_name, quantity_name):
    #Extracts data from CVS file at "location" and then sums over the desired quantity, given by "quantity_name" for each model iteration'
    #returns the mean and variance of the sum of the quantity

    df = pd.read_csv(location)

    # get all unique values of the iterations in file
    unique_iters = df[iter_name].unique()

    output_array = np.zeros(len(unique_iters))

    # loop through iterations and sum the deaths
    for i in unique_iters:
        iter_df = df.loc[df[iter_name] == i]
        quantity_total = iter_df[quantity_name].sum()  # Checked the sums using excel
        output_array[i] = quantity_total

    output_mean = output_array.mean()
    output_variance = output_array.var()

    return output_mean,output_variance


my_location = "output_prediction_simu_11-09-2020_18-57-41.txt"
my_iter_name = "iter" #name of column which stores iterations
my_quantity_name = " inc_death" #name of model quanitity to be summed

deaths_mean, deaths_variance = data_extraction_sum(my_location,my_iter_name,my_quantity_name)

print("mean deaths for this parameter choice is ", deaths_mean, " the varaince is ", deaths_variance)

