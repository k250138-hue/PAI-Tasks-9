import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cancer_dataset = load_breast_cancer()
features = cancer_dataset.data
labels = cancer_dataset.target

total_samples, total_features = features.shape
print("Total number of samples:", total_samples)
print("Total number of features:", total_features)

class_totals = np.bincount(labels)
for label_name, total in zip(cancer_dataset.target_names, class_totals):
    print(f"Target class '{label_name}': {total}")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

feature_scaler = StandardScaler()
X_train_norm = feature_scaler.fit_transform(X_train)
X_test_norm = feature_scaler.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train_norm, y_train)

accuracy = classifier.score(X_test_norm, y_test)
print(f"Model Test Accuracy: {accuracy:.4f}")

positive_probs = classifier.predict_proba(X_test_norm)[:, 1]

for cutoff in [0.3, 0.5, 0.7]:
    threshold_preds = (positive_probs >= cutoff).astype(int)
    positive_count = np.sum(threshold_preds)
    cutoff_acc = accuracy_score(y_test, threshold_preds)
    print(f"Cutoff {cutoff}: Class 1 Predictions = {positive_count:3d} | Accuracy = {cutoff_acc:.4f}")

def custom_sigmoid(val):
    return 1 / (1 + np.exp(-val))

decision_scores = classifier.decision_function(X_test_norm)
manual_probs = custom_sigmoid(decision_scores)

print("First 5 manual sigmoid outputs :", np.round(manual_probs[:5], 6))
print("First 5 predict_proba outputs  :", np.round(positive_probs[:5], 6))
print("Values match closely           :", np.allclose(manual_probs, positive_probs))