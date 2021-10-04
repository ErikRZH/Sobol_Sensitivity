'this class parses model runs and returns a CSV with the parameters and mean and variance of the output to be analysed'
import csv
import pandas as pd


location = "output_prediction_simu_11-09-2020_18-57-41.txt"
iter_name = "iter" #name of column which stores iterations
quantity_name = " inc_death" #name of model quanitity to be summed

df = pd.read_csv(location)

#get all unique values of the iterations in file
uniqueValues = df[iter_name].unique()

print(uniqueValues)

# loop through iterations
for i in uniqueValues:
    iter_df = df.loc[df[iter_name] == i]
    quantity_total = iter_df[quantity_name].sum()
    print(quantity_total) #Checked the sums using excel


print(df.head(3))
print(type(df))


