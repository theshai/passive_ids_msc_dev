import pandas as pd
import sklearn as sk

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
        return X_train, X_test
    
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

        print("\nSelected columns:")
        print(selected_columns)

        print("\nRemoved columns:")
        print(removed_columns)

        print("\nNumber of original features:", X_train.shape[1])
        print("Number of selected features:", len(selected_columns))
        print("Number of removed features:", len(removed_columns))
        
    elif method == "correlation":
        corr_matrix = pd.DataFrame(X_train).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_train_selected = pd.DataFrame(X_train).drop(columns=to_drop)
        X_test_selected = pd.DataFrame(X_test).drop(columns=to_drop)
        
    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_train, y_train)
        mi_scores_series = pd.Series(mi_scores, index=X_train.columns)
        selected_features = mi_scores_series[mi_scores_series > threshold].index
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
    elif method == "select_k_best":
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
    else:
        raise ValueError("Invalid feature selection method specified.")
    
    return X_train_selected, X_test_selected
 