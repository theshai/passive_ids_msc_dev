import sklearn as sk
from sklearn.metrics import (accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        confusion_matrix,
        classification_report
)     

def analize_model(X_train_scaled, X_test_scaled, y_train, y_test):                         

        # first model
        lr_model = sk.linear_model.LogisticRegression(max_iter=2000,random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        print("Logistic Regression model trained.....")

        y_pred = lr_model.predict(X_test_scaled)
        # for roc-auc, we need the predicted probabilities for the positive class
        y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Matthews Correlation Coefficient:", mcc)
        print("ROC-AUC:", roc_auc)

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))