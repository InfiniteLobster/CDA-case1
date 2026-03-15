#--------------------Libraries--------------------#
#for data handling
import pandas as pd
#for hydra configs
from omegaconf import DictConfig
#--------------------Code--------------------#
#loader function for training dataset
def load_data(data_path:str,target_col:str):
    #Loading datasets
    train_df = pd.read_csv(data_path)
    #Spliting features and target
    y = train_df[target_col]
    X = train_df.drop(columns=[target_col])
    #returnign output
    return y, X
#loading function for prediction dataset (goal of the project)
def load_data_predi(data_path):
    #Loading datasets
    predi_df = pd.read_csv(data_path)
    #Spliting features and target (not needed for prediction but keeping consistent code logic with train dataset loading function)
    X_predi = predi_df.iloc[:,:]
    #returnign output
    return X_predi  
#wrapper for loading via configs(hydra)
def load_data_cfg(cfg_dataset:DictConfig):
    #retriving parameters (informations) from  from subconfig (from main config, structured like that for logical orgnization of code and configs)
    data_path = cfg_dataset.path
    target_col = cfg_dataset.target_col
    num_prefix = cfg_dataset.num_prefix
    cat_prefix = cfg_dataset.cat_prefix
    #getting data
    y, X  = load_data(data_path, target_col)
    #getting list of which columns hold categorical and numerical data
    names_cat_cols = [col for col in X.columns if col.startswith(cat_prefix)]
    names_num_cols = [col for col in X.columns if col.startswith(num_prefix)]
    #returnign output
    return  y, X, names_num_cols, names_cat_cols