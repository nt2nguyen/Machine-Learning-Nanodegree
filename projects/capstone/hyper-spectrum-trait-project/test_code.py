import numpy as np
import os
import random
import pandas as pd

split_ratio = [.8,.2]


def get_train_test_split(data_file, trait, data_set_num):
	df = pd.read_csv(data_file)
	filtered_general = df[df["Wave_350"].notnull()].query('Wave_800>0.35').query('Wave_800<0.6')
	if (data_set_num == 1):
		filtered_general = filtered_general[filtered_general['Exp'] != 'CB_Mex']
	filtered_trait_NA = filtered_general[filtered_general[trait].notnull()]
	return filtered_trait_NA


df = get_train_test_split("../Mtrx-LMA-Narea-SPAD_forZH.csv","LMA_O",0)
print(df)

# print(data_values.dtypes.index)
