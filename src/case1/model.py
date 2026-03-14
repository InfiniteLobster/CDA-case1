#--------------------Libraries--------------------#

from omegaconf import DictConfig
#
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
#--------------------Code--------------------#
def get_model(model_name:str, params:DictConfig):
    #
    if model_name == "lasso":
        model = Lasso(**params)
    elif model_name == "ridge":
        #retrieving parameters from config
        alpha = params.get("alpha", 1.0) #default value for alpha(1.0) if not provided in params
        fit_intercept = params.get("fit_intercept", True)#default value for fit_intercept (true) if not provided in params
        #getting the model
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    elif model_name == "elasticnet":
        #retrieving parameters from config
        alpha = params.get("alpha", 1.0) #default value for alpha(1.0) if not provided in params
        l1_ratio = params.get("l1_ratio", 0.5) #default value for l1_ratio(0.5) if not provided in params
        max_iter = params.get("max_iter", 1000) #default value for max_iter(1000) if not provided in params
        fit_intercept = params.get("fit_intercept", True) #default value for fit_intercept (true) if not provided in params
        #getting the model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=fit_intercept)
    elif model_name == "linearregression":
        model = LinearRegression(**params)
    elif model_name == "randomforest":
        model = RandomForestRegressor(**params)
    elif model_name == "pcr":
        n_components = params.get("n_components",10)

        model = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("regression", LinearRegression())
        ])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    #returning output
    return model





