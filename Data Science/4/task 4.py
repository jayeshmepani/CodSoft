import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load the dataset
data = pd.read_csv(r"E:\project\Codsoft\Data Science\4\advertising.csv")

# Create a directory to store plots if it doesn't already exist
plot_dir = "plots_advertising"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Explore the data
print("First few rows of the dataset:")
print(data.head())
print("\nSummary statistics of the dataset:")
print(data.describe())
print(data.info())

# Check Unique Values for each variable.
print("\nUnique Values for each variable:")
for i in data.columns.tolist():
  print(i,":",data[i].nunique())

# Data Cleaning
# Check for null values
print("\nCheck for null values:")
print(data.isnull().sum()*100/data.shape[0]) # Print percentage of null values
print()

# Outlier Analysis
fig, axs = plt.subplots(3, figsize=(5, 5))
plt1 = sns.boxplot(data['TV'], ax=axs[0])
plt1.set_xlabel('TV')
plt2 = sns.boxplot(data['Newspaper'], ax=axs[1])
plt2.set_xlabel('Newspaper')
plt3 = sns.boxplot(data['Radio'], ax=axs[2])
plt3.set_xlabel('Radio')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "outlier_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# Exploratory Data Analysis (EDA)
# Univariate Analysis
plt.figure(figsize=(6, 4))
sns.boxplot(data['Sales'])
plt.xlabel('Sales')
plt.savefig(os.path.join(plot_dir, "univariate_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bivariate Analysis
sns.pairplot(data, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.xlabel('Advertising Spending')
plt.ylabel('Sales')
plt.savefig(os.path.join(plot_dir, "bivariate_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.xlabel('Features')
plt.ylabel('Features')
plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# Model Building
# Performing Simple Linear Regression
X = data['TV']
y = data['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Add a constant to X_train
X_train_sm = sm.add_constant(X_train)

# Fit the regression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()

# Print the parameters and summary statistics
print(lr.params)
print(lr.summary())

# Model Evaluation
# Residual analysis
y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)
fig = plt.figure(figsize=(6, 4))
sns.histplot(res, bins=15, kde=True)
fig.suptitle('Error Terms', fontsize=15)
plt.xlabel('Error', fontsize=15)
plt.savefig(os.path.join(plot_dir, "residual_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.scatter(X_train, res)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.savefig(os.path.join(plot_dir, "residual_vs_fitted.png"), dpi=300, bbox_inches='tight')
plt.close()

# Predictions on the Test Set
X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)

# RMSE and R-squared on the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R-squared:", r_squared)

# Visualize the fit on the test set
plt.scatter(X_test, y_test)
plt.plot(X_test, lr.params.iloc[0] + lr.params.iloc[1] * X_test, 'r')
plt.xlabel('TV Advertising ($)')
plt.ylabel('Sales (units)')
plt.savefig(os.path.join(plot_dir, "linear_regression_fit.png"), dpi=300, bbox_inches='tight')
plt.close()
