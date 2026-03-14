#--------------------Libraries--------------------#
#
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#--------------------Code--------------------#
#function to get preprocessor for numeric data with scaling
def get_numeric_scaled(strat:str):
    #building pipeline
    numeric_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy=strat)),
    ("scaler", StandardScaler())
    ])
    #returning output
    return numeric_scaled   
#function to get preprocessor for numeric data without scaling
def get_numeric_unscaled(strat:str):
    #building pipeline
    numeric_unscaled = Pipeline([
    ("imputer", SimpleImputer(strategy=strat))
    ])
    #returning output
    return numeric_unscaled 
#function to get preprocessor for categorical data
def get_categorical(strat:str):
    #building pipeline
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy=strat)),
        ("encoder", OneHotEncoder(handle_unknown = "ignore"))
    ])
    #returning output
    return categorical
#function to get preprrocessor ColumnTansformer
def get_preprocessor(flag_scale:bool, num_strat:str, cat_strat:str, names_num_cols:list, names_cat_cols:list):
    #gettign categorical preprocessor (no function difference depending on option,  no need for ifelse)
    categorical = get_categorical(cat_strat)
    #deciding which function to use for numeric preprocessor construction (based on flag_scale)
    if (flag_scale):
        #getting numeric preprocessor
        numeric = get_numeric_scaled(num_strat)
    else:
        #getting numeric preprocessor
        numeric = get_numeric_unscaled(num_strat)
    #constructing ColumnTransformer preprocessor
    preprocessor = ColumnTransformer([
        ("num", numeric, names_num_cols),
        ("cat", categorical, names_cat_cols)
    ])
    #returning output
    return preprocessor

     
