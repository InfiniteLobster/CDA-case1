#--------------------Libraries--------------------#
#for piepline construction
from sklearn.pipeline import Pipeline
#to get submodules (preprocessing and model) for better orgnization of code and logic
from preprocessing import get_preprocessor
from model import get_model
#--------------------Code--------------------#
#Function to build pipeline (joining pre-processing and model choices)
def build_pipeline(names_num_cols:list, names_cat_cols:list, flag_scale:bool, num_strat:str, cat_strat:str, model_name:str, model_params:dict):
    #getting preproccesor
    preprocessor = get_preprocessor(flag_scale, num_strat, cat_strat, names_num_cols, names_cat_cols)
    #getting model
    model = get_model(model_name, model_params)
    #building pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    #returning output
    return pipeline
#wrapper for pipeline building via configs(hydra)
def build_pipeline_cfg(names_num_cols, names_cat_cols,cfg_preprocess:dict, cfg_model:dict):
    #retriving parameters from preprocessing subconfig (from main config, structured like that for logical orgnization of code and configs)
    flag_scale = cfg_preprocess["flag_scale"]
    num_strat = cfg_preprocess["num_strat"]
    cat_strat = cfg_preprocess["cat_strat"]
    #retriving parameters (informations) of model from subconfig (from main config, structured like that for logical orgnization of code and configs)
    model_name = cfg_model["name"]
    model_params = cfg_model["params"]
    #building pipeline
    pipeline = build_pipeline(names_num_cols, names_cat_cols, flag_scale, num_strat, cat_strat, model_name, model_params)
    #returning output
    return pipeline