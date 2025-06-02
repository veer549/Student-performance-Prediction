import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df=pd.read_csv("student_performance_large_dataset.csv")
print(df)
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns.to_list())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv("cleaned_student_performance.csv", index=False)
df=pd.read_csv("cleaned_student_performance.csv")

print("Missing values after cleaning:")
print(df.isnull().sum())

#EDA analysis
# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Online_Courses_Completed'], bins=30, kde=True)
plt.title('Distribution of Online_Courses_Completed')
plt.xlabel('Online_Courses_Completed')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis
# Exam_Score vs Study_Hours_per_Week
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Study_Hours_per_Week', y='Exam_Score (%)', data=df)
plt.title('Exam Score vs Study Hours per Week')
plt.xlabel('Study Hours per Week')
plt.ylabel('Exam Score (%)')
plt.show()

#multivariate analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Study_Hours_per_Week',y='Exam_Score (%)', hue='Age', data=df)
plt.title('Relationship between Study Hours and Exam_Score (%) by Age')
plt.xlabel('Study_Hours_per_Week')
plt.ylabel('Exam_Score (%)')
plt.legend(title='Age')
plt.show()


#Linear Regression Model
X = df[['Study_Hours_per_Week', 'Online_Courses_Completed']]
y = df['Exam_Score (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"predicted values: {y_pred}")

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R2 Score:', r2)

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Exam Score (%)')
plt.ylabel('Predicted Exam Score (%)')
plt.title('Actual vs Predicted Exam Score (%)')
plt.legend()
plt.show()
