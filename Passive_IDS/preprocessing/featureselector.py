import pandas as pd
import sklearn as sk
import numpy as np

def apply_feature_selection(X_train,X_test,y_train,method="none",threshold=0.0,k=20):
    """
    Apply feature selection to the training and testing datasets.

    Parameters:
    - X_train: Training features (DataFrame)
    - X_test: Testing features (DataFrame)
    - y_train: Training labels (Series)
    - method: Feature selection method ("none", "variance", "correlation", "mutual_info", "select_k_best")
    - threshold: Threshold for variance or correlation methods
    - k: Number of top features to select for select_k_best method

    Returns:
    - X_train_selected: Training features after feature selection
    - X_test_selected: Testing features after feature selection
    """
    
    if method == "none":
        X_train_selected = X_train.copy()
        X_test_selected = X_test.copy()
        selected_columns=X_train.columns.tolist()
        removed_columns=[]
    
    elif method == "variance":
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        X_train_selected = selector.fit_transform(X_train)
        X_test_selected = selector.transform(X_test)
        # True = kept, False = removed
        support = selector.get_support()

        # Get kept columns
        selected_columns = X_train.columns[support].tolist()

        # Get removed columns
        removed_columns = X_train.columns[~support].tolist()
      
    elif method == "correlation":
        corr_matrix = pd.DataFrame(X_train).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_train_selected = pd.DataFrame(X_train).drop(columns=to_drop)
        X_test_selected = pd.DataFrame(X_test).drop(columns=to_drop)
        removed_columns = to_drop
        selected_columns = [column for column in X_train.columns if column not in to_drop]

    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_train, y_train)
        mi_scores_series = pd.Series(mi_scores, index=X_train.columns)
        selected_features = mi_scores_series[mi_scores_series > threshold].index
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        removed_columns = [column for column in X_train.columns if column not in selected_features]
        selected_columns = selected_features.tolist()
        
    elif method == "select_k_best":
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        removed_columns = [column for i, column in enumerate(X_train.columns) if not selector.get_support()[i]]
        selected_columns = [column for i, column in enumerate(X_train.columns) if selector.get_support()[i]]
    else:
        raise ValueError("Invalid feature selection method specified.")
    # maybe add anova test for feature selection, but for now, let's keep it simple and use the above methods.
    return X_train_selected, X_test_selected,selected_columns, removed_columns,method
 