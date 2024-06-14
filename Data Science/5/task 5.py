import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# Create a directory to store plots if it doesn't already exist
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Load the dataset
data = pd.read_csv(r"E:\project\Codsoft\Data Science\5\creditcard.csv")

# Explore the dataset
print(data.head())
print(data.isnull().sum())
print(data.describe())
print(data.info())

# Check class distribution
print(data['Class'].value_counts())

# Visualize class distribution
sns.countplot(data=data, x="Class")
plt.title("Countplot for Class")
plt.savefig(os.path.join(plot_dir, "Class_Countplot.png"), dpi=300, bbox_inches='tight')

plt.close()

# Visualize distribution of class with log scale
sns.histplot(data['Class'])
plt.yscale('log')
plt.savefig(os.path.join(plot_dir, "Class_Distribution_LogScale.png"), dpi=300, bbox_inches='tight')

plt.close()

# Display total amount for each class
print(data.groupby('Class')['Amount'].sum())

# Visualize distribution of transaction amount and time
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.histplot(amount_val, ax=ax[0], color='r', kde=True, stat='density')  # Set stat='density'
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.histplot(time_val, ax=ax[1], color='b', kde=True, stat='density')  # Set stat='density'
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
plt.savefig(os.path.join(plot_dir, 'Transaction_Time_Distribution.png'), dpi=300, bbox_inches='tight')

plt.close()

# Preprocess the data
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop('Class', axis=1)
y = data['Class']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=4),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'Neural Network': MLPClassifier()
}

# Train and evaluate models
results = {}
for name, model in tqdm(models.items(), desc='Training Models', unit='model'):
    model.fit(X_train_scaled, y_train)

    if isinstance(model, SVC) and not model.probability:
        y_pred_scores = model.decision_function(X_test_scaled)
        y_pred_proba = (y_pred_scores - y_pred_scores.min()) / (y_pred_scores.max() - y_pred_scores.min())
    else:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    results[name] = {'confusion_matrix': cm, 'precision_recall_auc': pr_auc}

# Visualize results
for name, result in tqdm(results.items(), desc='Visualizing Results', unit='model'):
    print(name)
    print("AUPRC:", result['precision_recall_auc'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print()

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))

    for name, result in tqdm(results.items(), desc='Visualizing Results', unit='model'):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f"{name} (AUPRC = {result['precision_recall_auc']:.2f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'Precision_Recall_Curve_All_Models.png'), dpi=300, bbox_inches='tight')
    
    plt.close()

# Model Accuracies
accuracies = {}
for name, model in tqdm(models.items(), desc='Calculating Accuracies', unit='model'):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy

print("Model Accuracies:")
for name, accuracy in accuracies.items():
    print(f"{name}: {accuracy:.4f}")

# Bar plot of Model Accuracies
plt.figure(figsize=(10, 6))
model_names = list(accuracies.keys())
accuracy_scores = list(accuracies.values())
plt.bar(model_names, accuracy_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0.998, 1)  # Set y-axis limits to better visualize differences
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(plot_dir, 'Model_Accuracies.png'), dpi=300, bbox_inches='tight')

plt.close()

# Add the provided functions to your code
def logic_regression(x_train, y_train, x_test):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_train_pred = lr.predict(x_train)
    y_train_cl_report = classification_report(y_train, y_train_pred, target_names=['No Fraud', 'Fraud'])

    print("_" * 100)
    print("TRAIN MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_train_cl_report)
    y_test_pred = lr.predict(x_test)
    y_test_cl_report = classification_report(y_test, y_test_pred, target_names=['No Fraud', 'Fraud'])

    print("_" * 100)
    print("TEST MODEL CLASSIFICATION REPORT")
    print("_" * 100)
    print(y_test_cl_report)
    print("_" * 100)

    return y_test_pred, lr


def conf_mat(y_test, y_test_pred, title=None, filepath=None):
    con_mat = confusion_matrix(y_test, y_test_pred)
    labels = ['No Fraud', 'Fraud']
    sns.heatmap(con_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if title:
        plt.title(title)
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()


def KNeighbors(x_train, y_train, x_test):
    Kneib = KNeighborsClassifier(n_neighbors=4)
    Kneib.fit(x_train, y_train)
    y_train_pred = Kneib.predict(x_train)
    y_train_cl_report = classification_report(y_train, y_train_pred, target_names=['No Fraud', 'Fraud'])

    print("_" * 50)
    print("TRAIN MODEL CLASSIFICATION REPORT")
    print("_" * 50)
    print(y_train_cl_report)
    y_test_pred = Kneib.predict(x_test)
    y_test_cl_report = classification_report(y_test, y_test_pred, target_names=['No Fraud', 'Fraud'])

    print("_" * 50)
    print("TEST MODEL CLASSIFICATION REPORT")
    print("_" * 50)
    print(y_test_cl_report)
    print("_" * 50)

    return y_test_pred, Kneib

# Train and evaluate logistic regression model
y_test_pred_lr, lr_model = logic_regression(X_train_scaled, y_train, X_test_scaled)

# Print and visualize classification report and confusion matrix for logistic regression
print("_" * 100)
print("Logistic Regression Results")
print("_" * 100)
print("TRAIN MODEL CLASSIFICATION REPORT")
print("_" * 100)
print(classification_report(y_train, lr_model.predict(X_train_scaled), target_names=['No Fraud', 'Fraud']))
print("_" * 100)
print("TEST MODEL CLASSIFICATION REPORT")
print("_" * 100)
print(classification_report(y_test, y_test_pred_lr, target_names=['No Fraud', 'Fraud']))
print("_" * 100)
print("Confusion Matrix for Logistic Regression")
print("_" * 100)
conf_mat(y_test, y_test_pred_lr, title='Confusion Matrix for Logistic Regression', filepath=os.path.join(plot_dir, 'Logistic_Regression_Confusion_Matrix.png'))

# Train and evaluate K-Nearest Neighbors model
y_test_pred_kneib, kneib_model = KNeighbors(X_train_scaled, y_train, X_test_scaled)

# Print and visualize classification report and confusion matrix for K-Nearest Neighbors
print("_" * 100)
print("K-Nearest Neighbors Results")
print("_" * 100)
print("TRAIN MODEL CLASSIFICATION REPORT")
print("_" * 100)
print(classification_report(y_train, kneib_model.predict(X_train_scaled), target_names=['No Fraud', 'Fraud']))
print("_" * 100)
print("TEST MODEL CLASSIFICATION REPORT")
print("_" * 100)
print(classification_report(y_test, y_test_pred_kneib, target_names=['No Fraud', 'Fraud']))
print("_" * 100)
print("Confusion Matrix for K-Nearest Neighbors")
print("_" * 100)
conf_mat(y_test, y_test_pred_kneib, title='Confusion Matrix for K-Nearest Neighbors', filepath=os.path.join(plot_dir, 'KNearest_Neighbors_Confusion_Matrix.png'))

# ROC Curve for Logistic Regression
lr_prob = lr_model.predict_proba(X_test_scaled)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob[:, 1], pos_label=1)

# ROC Curve for K-Nearest Neighbors
KNeib_prob = kneib_model.predict_proba(X_test_scaled)
fpr_kneib, tpr_kneib, _ = roc_curve(y_test, KNeib_prob[:, 1], pos_label=1)

# RANDOM
random_probs = [0 for _ in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# Combine the data from Logistic Regression, K-Nearest Neighbors, and RANDOM
plt.plot(fpr_lr, tpr_lr, linestyle='--', color='orange', label='Logistic Regression')
plt.plot(fpr_kneib, tpr_kneib, linestyle='-', color='green', label='K-Nearest Neighbors')
plt.plot(p_fpr, p_tpr, linestyle='dashdot', color='blue', label='RANDOM')

# Title
plt.title('ROC curve')
# X label
plt.xlabel('False Positive Rate')
# Y label
plt.ylabel('True Positive rate')

# Legend
plt.legend(loc='best')
# Save the figure
plt.savefig(os.path.join(plot_dir, 'ROC_combined.png'), dpi=300, bbox_inches='tight')
# Show the plot for a short time

plt.close()
