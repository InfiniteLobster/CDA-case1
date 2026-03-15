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
        #retrieving parameters from config (default values are set according to sklearn's default parameters for Lasso)
        alpha = params.get("alpha", 1.0) #default value for alpha(1.0) if not provided in params
        fit_intercept = params.get("fit_intercept", True)#default value for fit_intercept (true) if not provided in params
        max_iter = params.get("max_iter", 1000) #default value for max_iter(1000) if not provided in params
        #getting the model
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
    elif model_name == "ridge":
        #retrieving parameters from config (default values are set according to sklearn's default parameters for Ridge)
        alpha = params.get("alpha", 1.0) #default value for alpha(1.0) if not provided in params
        fit_intercept = params.get("fit_intercept", True)#default value for fit_intercept (true) if not provided in params
        max_iter = params.get("max_iter", 1000) #default value for max_iter(1000) if not provided in params
        #getting the model
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
    elif model_name == "elasticnet":
        #retrieving parameters from config (default values are set according to sklearn's default parameters for ElasticNet)
        alpha = params.get("alpha", 1.0) #default value for alpha(1.0) if not provided in params
        l1_ratio = params.get("l1_ratio", 0.5) #default value for l1_ratio(0.5) if not provided in params
        max_iter = params.get("max_iter", 1000) #default value for max_iter(1000) if not provided in params
        fit_intercept = params.get("fit_intercept", True) #default value for fit_intercept (true) if not provided in params
        #getting the model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=fit_intercept)
    elif model_name == "randomforest":
        #retrieving parameters from config (default values are set according to sklearn's default parameters for RandomForestRegressor)
        n_estimators = params.get("n_estimators", 100) 
        max_depth = params.get("max_depth", None) 
        min_samples_leaf = params.get("min_samples_leaf", 1) 
        min_samples_split = params.get("min_samples_split", 2) 
        max_features = params.get("max_features", 1.0) 
        random_state = params.get("random_state", None) 
        #getting the model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state, 
        )
    elif model_name == "pcr":
        #retrieving parameters from config (default values are set according to sklearn's default parameters for PCA)
        n_components = params.get("n_components",10)
        #getting the model
        model = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("regression", LinearRegression())
        ])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    #returning output
    return model





