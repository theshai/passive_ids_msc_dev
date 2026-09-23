import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
'''
note for future developer:
Add tuner per each new model!
'''

# for random forest
def tune_random_forest(X_train,y_train):

    model=RandomForestClassifier(
        random_state=42,
        n_jobs=1 #use one core....
    )
    #list of parameters to try (maybe change later)
    parameters = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"]
    }

    #let the search begin...
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=2, #parallel fit
        pre_dispatch=2, # prepare 2 jobs at a time
        verbose=2
    )

    search.fit (X_train,y_train)

    return(
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )

#for XGboost, same idea just seperated
def tune_xgboost(X_train, y_train):

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )

    # List of parameters to try
    parameters = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3]
    }

    # Let the search begin...
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=2,       # parallel fit
        pre_dispatch=2, # prepare 2 jobs at a time
        verbose=2
    )

    search.fit(
        X_train,
        y_train
    )

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )