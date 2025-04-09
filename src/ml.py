import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

# Scikit-Learn Libraries for Model Building and Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Scikit-Learn Libraries for Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Scikit-Learn Libraries for Evaluation Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score

# Suppress Convergence Warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

"""
Create Dataset (for demo purposes only)
"""

np.random.seed(0)

def generate_user_id(num_users):
    return lambda: str(np.random.randint(1, num_users))

def generate_login_time():
    return datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))

def generate_ip_address():
    return '.'.join([str(np.random.randint(0, 255)) for _ in range(4)])

def generate_device_type():
    devices = ["Windows", "Mac", "Linux", "Android", "iOS"]
    return random.choice(devices)

def generate_location():
    locations = ["Boston", "New York", "Atlanta", "Chicago", "Houston", "Phoenix", "San Francisco", "Seattle"]
    return random.choice(locations)

def generate_activity_type():
    activities = ["file_access", "database_query", "report_generation", "edit_template"]
    return random.choice(activities)

def generate_activity_period():
    return timedelta(minutes=random.randint(1, 180))

def generate_dataset(num_samples=1000, num_users=100):
    data = []
    col_field_pairs = [
        ('user_id', generate_user_id(num_users)),
        ('login_time', generate_login_time),
        ('ip_address', generate_ip_address),
        ('device_type', generate_device_type),
        ('location', generate_location),
        ('activity_type', generate_activity_type),
        ('activity_period', generate_activity_period),
    ]
    data = [[func() for _, func in col_field_pairs] for _ in range(num_samples)]
    return pd.DataFrame(data, columns=[col for col, _ in col_field_pairs])

# Generate dataset
num_samples = 20000
num_users = 200
df = generate_dataset(num_samples=num_samples, num_users=num_users)

# Add a target variable for classification (Password sharing prediction)
# Assume a user is marked as sharing (1) if their login_time not in (6, 23) and activity_period not in [10, 90]
df['sharing'] = np.where((df['login_time'].dt.hour <= 6) | (df['login_time'].dt.hour >= 23) | (df["activity_period"].dt.seconds // 60 < 10) | (df["activity_period"].dt.seconds // 60 > 90), 1, 0)

df.head()

"""
Data Preprocessing
"""

# Feature Engineering: Extract logging hour
df['login_hour'] = df['login_time'].dt.hour

# Feature Engineering: Whether on a weekday
df['is_weekday'] = df['login_time'].dt.dayofweek.apply(lambda x: 1 if x < 5 else 0)

# Feature Engineering: Login during working hours
df['is_working_hours'] = df['login_time'].dt.hour.apply(lambda x: 1 if 6 < x < 23 else 0)

# Feature Engineering: Extract activity hour
df["activity_hours"] = df["activity_period"].dt.seconds // 3600

# Feature Engineering: Change on device type
device_change_counts = df.groupby("user_id")["device_type"].nunique().reset_index(name="device_change_count")
df = pd.merge(df, device_change_counts, on="user_id", how="left")

# Feature Engineering: Change on activity type
activity_counts = df.groupby(["user_id", "activity_type"]).size().reset_index(name="activity_count")
df = pd.merge(df, activity_counts, on=["user_id", "activity_type"], how="left")

# Feature Engineering: Change on IP
df["ip_prefix"] = df["ip_address"].apply(lambda x: ".".join(x.split(".")[:2]))
ip_change_counts = df.groupby(["user_id"])["ip_prefix"].nunique().reset_index(name="ip_change_count")
df = pd.merge(df, ip_change_counts, on="user_id", how="left")

# One-Hot Encoding
df = pd.get_dummies(df, columns=['device_type', 'location', 'activity_type'], drop_first=True)

# Drop unnecessary columns
df.drop(columns=['user_id', 'login_time', 'ip_address', 'activity_period'], inplace=True)

df.head()

"""
Data Splitting
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X = df.drop(columns=['sharing'])  # Features
y = df['sharing']  # Target variable

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Cross-Validation
"""

# Define the Logistic Regression model
logreg = LogisticRegression(random_state=42)

# Define the cross-validation strategy (Stratified K-Fold for imbalanced classes)
logreg_scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')

print(f"Logistic Regression Cross-Validation Accuracy: {logreg_scores.mean():.4f}")

# Observe the 5 folds cross-validation scores
cross_val_scores = pd.DataFrame(logreg_scores)

plt.figure(figsize=(6, 3))
plt.bar(cross_val_scores.index, cross_val_scores[0])
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Cross-Validation Accuracy')

logreg = LogisticRegression(random_state=42, max_iter=1000)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the metrics we want to evaluate
scoring = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score),
    'roc_auc': 'roc_auc'  # Built-in scoring function for ROC-AUC
}

from sklearn.model_selection import cross_validate

# Perform cross-validation for all metrics defined in the scoring dictionary
multi_cv_results = cross_validate(logreg, X, y, cv=cv_strategy, scoring=scoring, return_train_score=True)

# Print results for each metric
for metric in scoring.keys():
    print(f"\n{metric.capitalize()} Scores: {multi_cv_results['test_' + metric]}")
    print(f"Mean {metric.capitalize()} Score: {np.mean(multi_cv_results['test_' + metric]):.4f}")
    print(f"Standard Deviation of {metric.capitalize()} Score: {np.std(multi_cv_results['test_' + metric]):.4f}")

"""
Model Training and Prediction
"""

# Step 1: Train a Logistic Regression model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)

# Step 2: Predict on the test set
y_pred_logreg = logreg.predict(X_test)

# Step 3: Evaluate the Logistic Regression model
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
logreg_report = classification_report(y_test, y_pred_logreg)
logreg_confusion_matrix = confusion_matrix(y_test, y_pred_logreg)

"""
Model Evaluation
"""

logreg_accuracy

print(logreg_confusion_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(logreg_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Sharing', 'Sharing'], yticklabels=['Not Sharing', 'Sharing'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

print(logreg_report)

from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for ROC/AUC
y_probs_logreg = logreg.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs_logreg)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

y_probs = logreg.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

"""
Hyperparameter Tuning
"""

from sklearn.model_selection import GridSearchCV

# Step1: Build an initial model (not tuned)
initial_tree = DecisionTreeClassifier(random_state=42)
initial_tree.fit(X_train, y_train)

# Evaluate the initial model on the test set to get a baseline performance
y_pred_initial = initial_tree.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred_initial)
print(f"Initial Model Accuracy (without tuning): {initial_accuracy:.4f}")

# Step 2: Hyperparameter Tuning using GridSearchCV (on the training set only)
# Define the parameter grid to search over
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Initialize GridSearchCV with a DecisionTreeClassifier
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='accuracy',
                           n_jobs=-1,  # Use all available cores
                           verbose=1)  # Print progress

# Fit GridSearchCV (only on the training set)
grid_search.fit(X_train, y_train)

# Get the best parameters and train the final model
print(f"Best Parameters from GridSearch: {grid_search.best_params_}")
best_tree = grid_search.best_estimator_  # Retrieve the best model

# Step 3: Evaluate the final model on the test set
y_pred_final = best_tree.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"Final Model Accuracy (after tuning): {final_accuracy:.4f}")

# Compare initial and final performance
print(f"Accuracy Improvement: {final_accuracy - initial_accuracy:.4f}")