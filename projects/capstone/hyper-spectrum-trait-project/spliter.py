import numpy as np
import os
import random
import pandas as pd

# 0.7 train, 0.1 validation and 0.2 test
split_ratio = [.7, .1, .2]


def get_train_val_test_split(data_file, trait, data_set_num):
    df = pd.read_csv(data_file)
    df['col_index'] = range(0, len(df))
    filtered_general = df[df["Wave_350"].notnull()].query('Wave_800>0.35').query('Wave_800<0.6')
    if (data_set_num == 1):
        filtered_general = filtered_general[filtered_general['Exp'] != 'CB_Mex']
    filtered_trait_NA = filtered_general[filtered_general[trait].notnull()]
    train=filtered_trait_NA.sample(frac=split_ratio[0])
    val = filtered_trait_NA.drop(train.index).sample(frac= (split_ratio[1] / (1 - split_ratio[0])))
    test=filtered_trait_NA.drop(train.index).drop(val.index)

    return train['col_index'].values, val['col_index'].values, test['col_index'].values



# Traits: LMA, Narea, SPAD, Nmass, Parea, Pmass 
DataFile0 = "data\Mtrx-LMA-Narea-SPAD_forZH.csv"

# Traits: Vcmax, Vcmax25, J, A (Photo_O), gs (Cond_O)
DataFile1 = "data\Out_Mtrx_Vcmax-Vc25-J_forZH.csv"

Traits0 = ["LMA_O", "Narea_O", "SPAD_O", "Nmass_O", "Parea_O", "Pmass_O"]
Traits1 = ["Vcmax", "Vcmax25", "J", "Photo_O", "Cond_O"] 

for data_set_num in range(0, 2):
    if (data_set_num == 0):
        current_trait_set = Traits0
        current_file = DataFile0
    else:
        current_trait_set = Traits1
        current_file = DataFile1

    for trait in current_trait_set:
        print("processing: ", current_file, trait, data_set_num)
        train, val, test = get_train_val_test_split(current_file, trait, data_set_num)
        print("train size: " + str(len(train)) + " val size:" + str(len(val)) + " test size:" + str(len(test)) )

        np.savez('data\splits\split_'+ trait +'.npz', train=train, val=val, test=test)



