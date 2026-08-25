import pandas as pd
import pytest

from preprocessing.featureselector import apply_feature_selection

def test_apply_feature_selection_variance():
    # Create a sample dataset
    data_train = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [1, 1, 1, 1, 1],  # Low variance feature
        'feature3': [5, 4, 3, 2, 1]
    }

    data_test = {
        'feature1': [1, 2],
        'feature2': [1, 1],  # Low variance feature
        'feature3': [5, 4]
    }

    X_train = pd.DataFrame(data_train)
    X_test = pd.DataFrame(data_test)
    y_train = pd.Series([0, 1, 0, 1, 0])

    # Apply feature selection with variance method
    X_train_selected, X_test_selected, selected_columns, removed_columns, method = apply_feature_selection(
        X_train,
        X_test,
        y_train,
        method="variance",
        threshold=0
    )

    # Check that the low variance feature is removed
    assert 'feature2' not in selected_columns
    assert 'feature2' in removed_columns
    assert X_train_selected.shape[1] == 2  # Only two features should remain