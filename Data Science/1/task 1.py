import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

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
