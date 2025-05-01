import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# print(df.head())


# print("Original Data: " ,df.head(10))
# print("\n")

# # Summary
# print("Descriptive Stats: \n",df.describe())
# print("\n")

# print("Info\n" ,df.info())
# print("\n")



# Data Cleaning
df = df.dropna()
print("Missing values: ",df.isnull().sum().sum())
print("\n")





# Objective 1

# Numerical Summary
numerical_cols = ['Run Time', 'Budget', 'Box Office', 'Nominations', 'Oscar Wins']
print("Descriptive Statistics:\n")
print(df[numerical_cols].describe())

# Categorical Summary
categorical_cols = ['Genre', 'Director', 'Studio', 'Country', 'Language', 'Certificate']
print("\nTop Categories:\n")
for col in categorical_cols:
    print(f"\n{col}:\n{df[col].value_counts().head(10)}")


# -------------------------------
# 2. Visualization - Categorical Distribution
# -------------------------------

# Set plot style
sns.set(style="whitegrid")

# Plot Top Genres
plt.figure(figsize=(10, 5))
top_genres = df['Genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title("Top 10 Movie Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# Plot Certificate Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, y='Certificate', order=df['Certificate'].value_counts().index, palette='coolwarm')
plt.title("Distribution of Movie Certificates")
plt.xlabel("Count")
plt.ylabel("Certificate")
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Runtime Distribution
# -------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df['Run Time'], bins=30, kde=True)
plt.title("Distribution of Movie Run Time")
plt.xlabel("Run Time (minutes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()






# Objective 2

# -------------------------------
# 1. Selecting Numerical Features
# -------------------------------
num_features = ['Run Time', 'Budget', 'Box Office', 'Nominations', 'Oscar Wins']

# Drop rows with missing values in these columns to calculate accurate correlations
df_corr = df[num_features].dropna()

# -------------------------------
# 2. Correlation Matrix
# -------------------------------
correlation_matrix = df_corr.corr()
print("Correlation Matrix:\n")
print(correlation_matrix)

# -------------------------------
# 3. Heatmap of Correlation
# -------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Movie Features")
plt.tight_layout()
plt.show()






# Objective 3

# -------------------------------
# 1. Extract Year from Release Date
# -------------------------------

df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Year'] = df['Release Date'].dt.year


# -------------------------------
# 2. Revenue and Budget Trends Over Time
# -------------------------------
yearly_data = df.groupby('Year')[['Budget', 'Box Office']].mean().dropna()

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_data)
plt.title('Average Budget and Box Office Revenue Over Time')
plt.xlabel('Year')
plt.ylabel('USD')
plt.tight_layout()
plt.show()


# -------------------------------
# 4. Trend in Oscar Wins Over Time
# -------------------------------
oscar_trend = df.groupby('Year')['Oscar Wins'].sum()

plt.figure(figsize=(12, 6))
sns.lineplot(x=oscar_trend.index, y=oscar_trend.values, color='gold')
plt.title('Total Oscar Wins Per Year')
plt.xlabel('Year')
plt.ylabel('Oscar Wins')
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Genre Popularity Over Time (Optional)
# -------------------------------
top_genres = df['Genre'].value_counts().nlargest(5).index
filtered = df[df['Genre'].isin(top_genres)]
genre_trend = filtered.groupby(['Year', 'Genre']).size().reset_index(name='Count')

plt.figure(figsize=(12, 6))
sns.lineplot(data=genre_trend, x='Year', y='Count', hue='Genre')
plt.title('Top Genre Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()






# Objective 4
# Convert Release Date to datetime and extract year
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Year'] = df['Release Date'].dt.year

# -------------------------------
# 1. Boxplots for Outlier Visualization
# -------------------------------
plt.figure(figsize=(14, 6))

# Budget
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Budget'], color='skyblue')
plt.title("Boxplot of Movie Budgets")

# Box Office
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Box Office'], color='lightgreen')
plt.title("Boxplot of Box Office Revenue")

plt.tight_layout()
plt.show()

# -------------------------------
# 2. Detect Outliers Using IQR Method
# -------------------------------
def detect_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers for Budget
budget_outliers, budget_lb, budget_ub = detect_outliers('Budget')
print(f"\n💰 Budget Outliers: {len(budget_outliers)} found")
print(budget_outliers[['Title', 'Budget']].sort_values(by='Budget', ascending=False).head())

# Detect outliers for Box Office
revenue_outliers, revenue_lb, revenue_ub = detect_outliers('Box Office')
print(f"\n🎥 Revenue Outliers: {len(revenue_outliers)} found")
print(revenue_outliers[['Title', 'Box Office']].sort_values(by='Box Office', ascending=False).head())






# objective 5

df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Year'] = df['Release Date'].dt.year

# -------------------------------
# 1. Success Metric Trends Over Time
# -------------------------------

# Grouping by year to get average success factors
yearly_success = df.groupby('Year')[['Box Office', 'Budget', 'Nominations', 'Oscar Wins']].mean().dropna()

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_success)
plt.title('Average Success Factors of Movies Over Time')
plt.xlabel('Year')
plt.ylabel('Average Value')
plt.tight_layout()
plt.show()

# -------------------------------
# 2. Success Ratio: Revenue to Budget Over Time
# -------------------------------
df['Success Ratio'] = df['Box Office'] / df['Budget']
success_ratio_yearly = df.groupby('Year')['Success Ratio'].mean().dropna()

plt.figure(figsize=(12, 5))
sns.lineplot(x=success_ratio_yearly.index, y=success_ratio_yearly.values, color='purple')
plt.title('Average Revenue-to-Budget Ratio Over Time')
plt.xlabel('Year')
plt.ylabel('Success Ratio')
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Top Years for Oscar Wins
# -------------------------------
oscar_totals = df.groupby('Year')['Oscar Wins'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=oscar_totals.index, y=oscar_totals.values, color='gold')
plt.title('Top 10 Years with Most Oscar Wins')
plt.xlabel('Year')
plt.ylabel('Total Oscar Wins')
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Scatter: Budget vs. Box Office
# -------------------------------
import numpy as np

# Create a binary column: Did the movie win at least 1 Oscar?
df['Won Oscar'] = df['Oscar Wins'].apply(lambda x: 'Yes' if x > 0 else 'No')

# Scatter Plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Convert Release Date to datetime and extract year
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Year'] = df['Release Date'].dt.year

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Budget',
    y='Box Office',
    size='Nominations',       # Size by number of nominations
    sizes=(40, 400),
    color='steelblue',        # Single uniform color
    alpha=0.7,
    edgecolor='black'
)

plt.title('Box Office vs. Budget\nPoint Size = Number of Nominations', fontsize=14)
plt.xlabel('Budget')
plt.ylabel('Box Office')

# Linear scales used here (default)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Nominations')
plt.show()
