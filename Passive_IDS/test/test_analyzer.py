import numpy as np
import pandas as pd
import pytest as pt

from preprocessing.analyzer import (calculate_metrics,analize_model_logostic_regression,analize_model_xgboost,analize_model_random_forest)

def test_calculate_metrics():
    # Test the calculate_metrics function with sample data
    y_test = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_pred_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9])

    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    assert metrics[0] == pt.approx(0.8) #accuracy
    assert metrics[1] == pt.approx(1.0) #precision
    assert metrics[2] == pt.approx(0.6666667) #recall
    assert metrics[3] == pt.approx(0.8) #f1
    assert metrics[4] == pt.approx(0.6666667) #mcc
    assert metrics[5] == pt.approx(1.0) #roc_auc

def test_analize_model_random_forest():
    # Test the analize_model_xgboost function with sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    metrics = analize_model_random_forest(X_train, X_test, y_train, y_test)
    assert isinstance(metrics, tuple)
    assert len(metrics) == 7  # Ensure that all metrics are returned
    #Get specific metrics
    accuracy, precision, recall, f1, mcc, roc_auc,fpr = metrics
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert -1.0 <= mcc <= 1.0
    assert 0.0 <= roc_auc <= 1.0

def test_analize_model_xgboost():
    # Test the analize_model_xgboost function with sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    metrics = analize_model_xgboost(X_train, X_test, y_train, y_test)
    assert isinstance(metrics, tuple)
    assert len(metrics) == 7  # Ensure that all metrics are returned
    #Get specific metrics
    accuracy, precision, recall, f1, mcc, roc_auc,fpr = metrics
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert -1.0 <= mcc <= 1.0
    assert 0.0 <= roc_auc <= 1.0

def test_analize_model_logistic_regression():
    # Test the analize_model_regression function with sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    metrics = analize_model_logostic_regression(X_train, X_test, y_train, y_test)
    assert isinstance(metrics, tuple)
    assert len(metrics) == 7  # Ensure that all metrics are returned
    #Get specific metrics
    accuracy, precision, recall, f1, mcc, roc_auc,fpr = metrics
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert -1.0 <= mcc <= 1.0
    assert 0.0 <= roc_auc <= 1.0
