import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, f1_score, roc_curve, confusion_matrix, fbeta_score
import numpy as np


# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = "E:\\project\\Codsoft\\Data Science\\1\\Titanic-Dataset.csv"
full_data = pd.read_csv(file_path)

# Explore the dataset
print(full_data.head())
print(full_data.isnull().sum())
print(full_data.describe())
print(full_data.info())

# Set the default figure size
plt.rcParams["figure.figsize"] = [10, 5]

# Function to save the plot as PNG
def save_plot_as_png(filename):
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
    plt.close()

# Create a directory to store plots if it doesn't already exist
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Explore the shape and first few rows of the dataset
print("Shape of the dataset:", full_data.shape)
print("First few rows of the dataset:\n", full_data.head())

# Histogram of 'Parch'
sns.histplot(full_data['Parch'], kde=False)
save_plot_as_png(os.path.join(plot_dir, "Parch_hist"))

# Distribution plot of 'Age' without histogram
sns.kdeplot(full_data['Age'], shade=True)
save_plot_as_png(os.path.join(plot_dir, "Age_kde"))

# Distribution plot of 'Age' with histogram
sns.histplot(full_data['Age'], kde=True)
save_plot_as_png(os.path.join(plot_dir, "Age_hist_with_kde"))

# Line plot of 'Age' vs. 'Fare' with 'Pclass' and 'Sex' as facets
sns.relplot(x="Age", y="Fare", col="Pclass", hue="Sex", style="Sex", kind="line", data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Age_vs_Fare_lineplot"))

# Scatter plot of 'Age' vs. 'Fare' colored by 'Sex'
sns.scatterplot(x="Age", y="Fare", hue="Sex", data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Age_vs_Fare_scatterplot"))

# Line plot of 'Age' vs. 'Fare' with 'Sex' as hue
sns.lineplot(x="Age", y="Fare", hue="Sex", style="Sex", data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Age_vs_Fare_lineplot_hue_Sex"))

# Bar plot of 'Survived' vs. 'Sex' with 'Pclass' as hue
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Survived_vs_Sex_barplot"))

# Strip plot of 'Sex' vs. 'Age' with 'Sex' as hue
sns.stripplot(x="Sex", y="Age", hue='Sex', data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Sex_vs_Age_stripplot"))

# Swarm plot of 'Sex' vs. 'Age' with 'Sex' as hue
sns.swarmplot(x="Sex", y="Age", hue='Sex', data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Sex_vs_Age_swarmplot"))

# Box plot of 'Survived' vs. 'Age'
sns.boxplot(x="Survived", y="Age", data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Survived_vs_Age_boxplot"))

# Violin plot of 'Survived' vs. 'Age' with 'Sex' as hue
sns.violinplot(x="Survived", y="Age", hue='Sex', data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Survived_vs_Age_violinplot"))

# Count plot of 'Survived'
sns.countplot(x="Survived", data=full_data, palette="Blues")
save_plot_as_png(os.path.join(plot_dir, "Survived_countplot"))

# Point plot of 'Sex' vs. 'Survived' with 'Pclass' as hue
sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Sex_vs_Survived_pointplot"))

# Regression plot of 'Age' vs. 'Fare'
sns.regplot(x="Age", y="Fare", data=full_data)
save_plot_as_png(os.path.join(plot_dir, "Age_vs_Fare_regplot"))

# Heatmap of correlation matrix (excluding non-numeric columns)
plt.figure(figsize=(10, 10))
columns_to_exclude = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
numeric_data = full_data.drop(columns=columns_to_exclude)
sns.heatmap(numeric_data.corr(), cmap="YlGnBu", annot=True, fmt=".2f")
save_plot_as_png(os.path.join(plot_dir, "correlation_heatmap"))

# FacetGrid of 'Survived' vs. 'Age' with 'Pclass' as row and column
g = sns.FacetGrid(full_data, col='Survived', row='Pclass')
g.map(plt.hist, 'Age')
g.add_legend()
plt.savefig(os.path.join(plot_dir, "Survived_vs_Age_FacetGrid.png"), dpi=300, bbox_inches='tight')
plt.close()

# Preprocessing
full_data['Age'].fillna(full_data['Age'].median(), inplace=True)
full_data['Fare'].fillna(full_data['Fare'].median(), inplace=True)
full_data['Embarked'].fillna(full_data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numeric
full_data['Sex'] = full_data['Sex'].map({'male': 0, 'female': 1})
full_data = pd.get_dummies(full_data, columns=['Embarked'], drop_first=True)

# Split the data into training and testing sets
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = full_data[features]
y_classification = full_data['Survived']
y_regression = full_data['Fare']

X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X, y_classification, test_size=0.2, random_state=42)
# X_train_regression, X_test_regression, y_train_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)

print("Shape of X:", X.shape)
print("Shape of y_regression:", y_regression.shape)
# X_train_regression, X_test_regression, y_train_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)
# Split the data into training and testing sets for regression
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)


# Scale the features
scaler = StandardScaler()
X_train_classification = scaler.fit_transform(X_train_classification)
X_test_classification = scaler.transform(X_test_classification)
X_train_regression = scaler.fit_transform(X_train_regression)
X_test_regression = scaler.transform(X_test_regression)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_classification, y_train_classification)
y_pred_knn = knn.predict(X_test_classification)
print("KNN Accuracy:", accuracy_score(y_test_classification, y_pred_knn))
print("KNN F1 Score:", f1_score(y_test_classification, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test_classification, y_pred_knn))

# Calculate additional metrics for KNN
mse_knn = mean_squared_error(y_test_classification, y_pred_knn)
f2_knn = fbeta_score(y_test_classification, y_pred_knn, beta=2)
r2_knn = r2_score(y_test_classification, y_pred_knn)

# Confusion Matrix for KNN
conf_matrix_knn = confusion_matrix(y_test_classification, y_pred_knn)
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for KNN")
save_plot_as_png(os.path.join(plot_dir, "KNN_confusion_matrix"))

# ROC Curve for KNN
fpr_knn, tpr_knn, _ = roc_curve(y_test_classification, knn.predict_proba(X_test_classification)[:, 1])
plt.plot(fpr_knn, tpr_knn, label="KNN")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for KNN")
plt.legend(loc="best")
save_plot_as_png(os.path.join(plot_dir, "KNN_roc_curve"))

# Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_classification, y_train_classification)
y_pred_log_reg = log_reg.predict(X_test_classification)
print("Logistic Regression Accuracy:", accuracy_score(y_test_classification, y_pred_log_reg))
print("Logistic Regression F1 Score:", f1_score(y_test_classification, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test_classification, y_pred_log_reg))

# Calculate additional metrics for Logistic Regression
mse_log_reg = mean_squared_error(y_test_classification, y_pred_log_reg)
f2_log_reg = fbeta_score(y_test_classification, y_pred_log_reg, beta=2)
r2_log_reg = r2_score(y_test_classification, y_pred_log_reg)

# Confusion Matrix for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_test_classification, y_pred_log_reg)
sns.heatmap(conf_matrix_log_reg, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Logistic Regression")
save_plot_as_png(os.path.join(plot_dir, "Logistic_Regression_confusion_matrix"))

# ROC Curve for Logistic Regression
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test_classification, log_reg.predict_proba(X_test_classification)[:, 1])
plt.plot(fpr_log_reg, tpr_log_reg, label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc="best")
save_plot_as_png(os.path.join(plot_dir, "Logistic_Regression_roc_curve"))

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_regression, y_train_regression)
y_pred_lin_reg = lin_reg.predict(X_test_regression)
print("Linear Regression MSE:", mean_squared_error(y_test_regression, y_pred_lin_reg))
print("Linear Regression R2 Score:", r2_score(y_test_regression, y_pred_lin_reg))

# Scatter plot for Linear Regression
plt.scatter(y_test_regression, y_pred_lin_reg)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: True vs Predicted Values")
save_plot_as_png(os.path.join(plot_dir, "Linear_Regression_true_vs_pred"))

# Residual plot for Linear Regression
residuals = y_test_regression - y_pred_lin_reg
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.title("Linear Regression Residuals")
save_plot_as_png(os.path.join(plot_dir, "Linear_Regression_residuals"))

# Regression plot for Linear Regression
sns.regplot(x=y_test_regression, y=y_pred_lin_reg, scatter_kws={"s": 10})
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: True vs Predicted Values (with regression line)")
save_plot_as_png(os.path.join(plot_dir, "Linear_Regression_true_vs_pred_with_regression_line"))

# Create a summary table
summary_table = pd.DataFrame({
    "Model": ["K-Nearest Neighbors (KNN)", "Logistic Regression", "Linear Regression"],
    "MSE": [mse_knn, mse_log_reg, mean_squared_error(y_test_regression, y_pred_lin_reg)],
    "F2 Score": [f2_knn, f2_log_reg, "N/A"],
    "R2 Score": [r2_knn, r2_log_reg, r2_score(y_test_regression, y_pred_lin_reg)]
})

print("\nSummary Table:")
print(summary_table)

# Save the summary table as a PNG
plt.figure(figsize=(8, 4))
plt.table(cellText=summary_table.values, colLabels=summary_table.columns, cellLoc='center', loc='center')
plt.axis('off')
save_plot_as_png(os.path.join(plot_dir, "Summary_Table"))
