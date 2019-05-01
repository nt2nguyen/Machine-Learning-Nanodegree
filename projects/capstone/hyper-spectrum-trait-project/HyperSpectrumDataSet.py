from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import pandas as pd


Traits0 = ["LMA_O", "Narea_O", "SPAD_O", "Nmass_O", "Parea_O", "Pmass_O"]
Traits1 = ["Vcmax", "Vcmax25", "J", "Photo_O", "Cond_O"] 



# Traits: LMA, Narea, SPAD, Nmass, Parea, Pmass 
DataFile0 = "data/Mtrx-LMA-Narea-SPAD_forZH.csv"

# Traits: Vcmax, Vcmax25, J, A (Photo_O), gs (Cond_O)
DataFile1 = "data/Out_Mtrx_Vcmax-Vc25-J_forZH.csv"

class HyperSpectrumDataSet(Dataset):
    """HyperSpectrum dataset."""

    def __init__(self, trait, split, drop_percent=0.0):
        """
        Args:
            trait: "LMA_O", "Narea_O", "SPAD_O", "Nmass_O", "Parea_O", "Pmass_O",
                    "Vcmax", "Vcmax25", "J", "Photo_O", "Cond_O"
            split: "train", "val", "test"
        """
        if trait in Traits0:
            data = pd.read_csv(DataFile0)
        elif trait in Traits1:
            data = pd.read_csv(DataFile1)
        else:
            print("The trait is not found")

        split_indexes = np.load('data/splits/split_'+ trait +'.npz')

        if split == "train":
            loaded_split_indexes = split_indexes['train']
        elif split == "val":
            loaded_split_indexes = split_indexes['val']
        elif split == "test":
            loaded_split_indexes = split_indexes['test']
        else:
            print("Need to specify train, val or test")

        train_df = data.loc[split_indexes['train'],[trait]]
        if drop_percent > 0.0:
            train_df=train_df.sample(frac=(1-drop_percent))
            
        trait_average = train_df.mean().values.astype(np.float32)
        trait_average = np.asscalar(trait_average)
        trait_std = train_df.std().values.astype(np.float32)
        trait_std = np.asscalar(trait_std)

        self.data_values = data.loc[loaded_split_indexes,'Wave_350':]
        self.data_labels = data.loc[loaded_split_indexes,[trait]] - trait_average
        self.data_labels = self.data_labels / trait_std

    def __len__(self):
        return len(self.data_values)

    def __getitem__(self, idx):
        return self.data_values.iloc[idx,:].values.astype(np.float32), self.data_labels.iloc[idx,0].astype(np.float32)
    


