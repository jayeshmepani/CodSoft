import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load the dataset
file_path = "E:\\project\\Codsoft\\Data Science\\2\\IMDb Movies India.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Explore the dataset
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.info())

# Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Data cleaning
df["Year"] = df["Year"].str.strip("()").astype(int)
df["Duration"] = df["Duration"].str.replace(" min", "").astype(int)
df["Votes"] = df["Votes"].str.replace(",", "").astype(int)

# Feature engineering
df['Genre_encoded'] = round(df.groupby('Genre')['Rating'].transform('mean'), 1)
df['Votes_encoded'] = round(df.groupby('Votes')['Rating'].transform('mean'), 1)
df['Director_encoded'] = round(df.groupby('Director')['Rating'].transform('mean'), 1)
df['Actor_1_encoded'] = round(df.groupby('Actor 1')['Rating'].transform('mean'), 1)
df['Actor_2_encoded'] = round(df.groupby('Actor 2')['Rating'].transform('mean'), 1)
df['Actor_3_encoded'] = round(df.groupby('Actor 3')['Rating'].transform('mean'), 1)

# Create a directory to store plots if it doesn't already exist
plot_dir = "plots_imdb"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Visualization before dropping 'Director' & 'Actor 1'
plt.figure(figsize=(16,6))
df["Director"].value_counts(ascending=False)[:15].plot(kind="bar")
plt.xticks(rotation=90)
plt.title("Director with no of movies ")
plt.savefig(os.path.join(plot_dir, "director_count.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

plt.figure(figsize=(16,6))
df["Actor 1"].value_counts(ascending=False)[:15].plot(kind="bar")
plt.xticks(rotation=90)
plt.title("Popular Actor ")
plt.savefig(os.path.join(plot_dir, "popular_actor.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Now you can drop the 'Director' column
df.drop(columns=['Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Name'], inplace=True)
df['Rating'] = round(df['Rating'], 1)

# Visualization
sns.pairplot(df)
plt.savefig(os.path.join(plot_dir, "pairplot.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Univariate analysis
plt.figure(figsize=(10, 6))
sns.histplot(df["Year"], kde=True)
plt.title("Distribution of years")
plt.savefig(os.path.join(plot_dir, "year_distribution.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Distribution of movie ratings
plt.figure(figsize=(10, 6))
sns.histplot(df["Rating"], kde=True)
plt.title("Distribution of movie ratings")
plt.savefig(os.path.join(plot_dir, "rating_distribution.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Scatter plot to examine if genre affects the rating
plt.figure(figsize=(10, 6))
sns.scatterplot(y="Genre_encoded", x="Rating", data=df)
plt.title("Does genre affect the rating?")
plt.savefig(os.path.join(plot_dir, "genre_vs_rating.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Scatter plot to examine how votes affect the rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Votes_encoded", y="Rating", data=df)
plt.title("How votes affect the rating?")
plt.savefig(os.path.join(plot_dir, "votes_vs_rating.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Count plot to visualize the distribution of movies over the years
plt.figure(figsize=(15, 6))
sns.countplot(x=df["Year"])
plt.xticks(rotation=90)
plt.savefig(os.path.join(plot_dir, "year_countplot.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", fmt=".2f", annot=True)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
#plt.pause(1)
#plt.close()

# Top 10 analysis
top_10_ratings = df[['Genre_encoded', 'Rating']].sort_values(by='Rating', ascending=False).head(10)
print("Top 10 movies based on ratings:")
print(top_10_ratings)

top_10_votes = df[['Genre_encoded', 'Votes_encoded']].sort_values(by='Votes_encoded', ascending=False).head(10)
print("\nTop 10 movies based on votes:")
print(top_10_votes)

# Model training and evaluation
x = df.drop("Rating", axis=1)
y = df["Rating"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Machine": SVR(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"MSE: {mse}")
    print(f"R^2 Score: {r2}")
