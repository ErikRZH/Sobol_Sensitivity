"""this class parses model runs and returns a CSV with the parameters and mean and variance of the output to be analysed"""
import csv
import os
import re
import pandas as pd
import numpy as np


def data_extraction_sum(location, iter_name, quantity_name):
    """
    Extracts data from CSV file at "location" and then sums over the desired quantity, given by "quantity_name" for each model iteration.
    returns: the mean and variance of the sum of the quantity.
    """

    df = pd.read_csv(location)

    # get all unique values of the iterations in file
    unique_iters = df[iter_name].unique()

    # initialising array for mean and variance calculation
    output_array = np.zeros(len(unique_iters))

    # loop through iterations and sum the deaths
    for i in unique_iters:
        iter_df = df.loc[df[iter_name] == i]
        quantity_total = iter_df[quantity_name].sum()  # sum of the quantity over the model run
        output_array[i] = quantity_total

    output_mean = output_array.mean()
    output_variance = output_array.var()

    return output_mean, output_variance


def get_index_locations():
    """
    Heavily data structure dependent function.
    Gives id and associated location of all runs as a dictionary.
    returns: dictionary with key=run_index,value=run_location.
    """

    # nr of different folders (1,2,3,4)
    folder_count = range(1, 4 + 1)
    folder_name_style = "UQ_eera_2020-09-11_part"
    all_folder_names = [folder_name_style + str(i) for i in folder_count]

    # nr of different runs
    run_count = range(160)
    run_name_style = "output_row_"
    all_run_names = [run_name_style + str(i) for i in run_count]

    path = os.getcwd()

    output_index_location = {}

    for current_folder in all_folder_names:

        runs_in_folder = [i for i in all_run_names if i in os.listdir(current_folder)]

        for run in runs_in_folder:

            run_location = os.path.join(path, current_folder, run)
            run_location_contents = os.listdir(run_location)

            # identifying data file using regex
            pattern = re.compile('^output_prediction_simu_')
            run_matches = list(filter(pattern.match, run_location_contents))

            if len(run_matches) > 1:
                raise ValueError(
                    "Multiple run files, " + "(output_prediction_simu_ ...)" + " in same folder: " + run_location)
            else:
                run_file = os.path.join(run_location, run_matches[0])
                run_index = re.match('.*?([0-9]+)$', run).group(1)  # get index from end of file
                output_index_location[int(run_index)] = run_file

    return output_index_location


def esoteric_data_to_CSV(output_csv_name):
    """

    Extracts model output data stored in an esoteric folder structure:
    ├───UQ_eera_2020-09-11_part1
    │   ├───output_row_0
    │   │    └───output_prediction_simu_...
    │   ├───output_row_1
    │   │    └───output_prediction_simu_...
    │   ...
    ...

    Where, "output_prediction_simu..." is a csv containing the output of 1000 different 200 day runs for 1 choice of parameters

    the parameters associated with the runs are located in: "posterior_parameters.csv".

    This function parses through this data and extracts the mean and variance of a quantity of interest and then writes:
        run index, parameters for run, mean of the quanitity of interest, variance of the quantity of interest
    to a CSV file.

    The CSV file this function creates is then of the type that it may be analysed using sensitvity_analysis_sobol.py

    Args:
        output_csv_name: the name of the parameters-output CSV file.
    """
    # create index-file location dictionary
    index_locations = get_index_locations()

    my_iter_name = "iter"  # name of column which stores iterations
    my_quantity_name = " inc_death"  # name of model quanitity to be summed

    # creating master CSV file
    quantity_mean = "total_deaths_mean"
    quantity_variance = "total_deaths_variance"
    df = pd.read_csv("posterior_parameters.csv")
    df.set_index("Index")
    df[quantity_mean] = np.nan
    df[quantity_variance] = np.nan

    # loop over indivies to make CSV file
    for i in df["Index"]:
        i_deaths_mean, i_deaths_variance = data_extraction_sum(index_locations[i], my_iter_name, my_quantity_name)
        df.at[i, quantity_mean] = i_deaths_mean
        df.at[i, quantity_variance] = i_deaths_variance
        print(i)

    df.to_csv(output_csv_name, index=False)
