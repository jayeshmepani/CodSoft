import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import os

# Load the dataset from CSV file
iris_data = pd.read_csv(r"E:\project\Codsoft\Data Science\3\IRIS.csv")

# Create a directory to store plots if it doesn't already exist
plot_dir = "plots_iris"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Explore the data
print("First few rows of the dataset:")
print(iris_data.head())
print(iris_data.isnull().sum())
print("\nSummary statistics of the dataset:")
print(iris_data.describe())
print("\nTarget Labels:", iris_data["species"].unique())
print(iris_data.info())

# Check Unique Values for each variable.
for i in iris_data.columns.tolist():
  print("Unique values in",i,":",iris_data[i].nunique())

# Plotting the distributions of columns
cols_with_missing_values = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
fig, axes = plt.subplots(len(cols_with_missing_values), 1, figsize=(10, 15))
fig.tight_layout(pad=5.0)

for i, col in enumerate(cols_with_missing_values):
    sns.histplot(iris_data[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
plt.savefig(os.path.join(plot_dir, "distributions_of_columns.png"), dpi=300, bbox_inches='tight')
plt.close()

# Species count plot
species_counts = iris_data['species'].value_counts()
plt.figure(figsize=(8, 8))
species_counts.plot(kind='pie', autopct='%1.2f%%', startangle=90)
plt.title('Species')
plt.legend(loc='upper left', labels=species_counts.index)
plt.savefig(os.path.join(plot_dir, "species_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot for sepal length vs sepal width
sns.lmplot(
    x="sepal_length",
    y="sepal_width",
    hue="species",
    palette="bright",
    data=iris_data,
    height=5.3,
    aspect=1.7
)
plt.title("Sepal Length VS Sepal Width")
plt.savefig(os.path.join(plot_dir, "sepal_length_vs_sepal_width_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot for petal length vs petal width
sns.lmplot(
    x="petal_length",
    y="petal_width",
    hue="species",
    palette="bright",
    data=iris_data,
    height=5.3,
    aspect=1.7
)
plt.title("Petal Length VS Petal Width")
plt.savefig(os.path.join(plot_dir, "petal_length_vs_petal_width_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

# Split the data into training and testing sets
X = iris_data.drop('species', axis=1)  # Features (sepal length, sepal width, petal length, petal width)
y = iris_data['species']  # Target variable (species)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=iris_data['species'].unique())
print("\nRandom Forest Model Performance:")
print("Accuracy:", rf_accuracy)
print("Classification Report:")
print(classification_rep)

# Visualize confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Random Forest Confusion Matrix')
plt.savefig(os.path.join(plot_dir, "rf_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# Train and evaluate KNN model with hyperparameter tuning
def evaluate_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(classification_report(y_val, y_pred_val, target_names=iris_data['species'].unique()))
    return acc_train, acc_val

n_neighbors_values = list(range(1, 15))
acc_train_values = []
acc_val_values = []

for n in n_neighbors_values:
    knn_model = KNeighborsClassifier(n_neighbors=n)
    acc_train, acc_val = evaluate_model(knn_model, X_train, y_train, X_test, y_test)
    acc_train_values.append(acc_train)
    acc_val_values.append(acc_val)

results_n_neighbors = pd.DataFrame({
    'n_neighbors': n_neighbors_values,
    'acc_train': acc_train_values,
    'acc_val': acc_val_values,
})

# Plotting accuracy values for different n_neighbors
plt.figure(figsize=(10, 5))
plt.plot(n_neighbors_values, acc_train_values, label='Train Accuracy')
plt.plot(n_neighbors_values, acc_val_values, label='Validation Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Hyperparameter Tuning')
plt.legend()
plt.savefig(os.path.join(plot_dir, "knn_hyperparameter_tuning.png"), dpi=300, bbox_inches='tight')
plt.close()

# Final KNN model evaluation with n_neighbors=10
knn_final_model = KNeighborsClassifier(n_neighbors=10)
knn_final_model.fit(X_train, y_train)
knn_y_pred = knn_final_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("\nKNN Model Performance with n_neighbors=10:")
print("Accuracy:", knn_accuracy)
print("Classification Report:")
print(classification_report(y_test, knn_y_pred, target_names=iris_data['species'].unique()))

# Visualize confusion matrix for KNN
cm_knn = confusion_matrix(y_test, knn_y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('KNN Confusion Matrix')
plt.savefig(os.path.join(plot_dir, "knn_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
