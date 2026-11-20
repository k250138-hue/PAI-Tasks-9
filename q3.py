import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

years_exp = [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10, 3, 5]
degree_level = ['BS', 'BS', 'MS', 'BS', 'MS', 'PhD', 'MS', 'PhD', 'BS', 'PhD', 'BS', 'MS']
annual_salary_k = [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88]

employee_df = pd.DataFrame({
    'YearsExp': years_exp,
    'Degree': degree_level,
    'AnnualSalary': annual_salary_k
})

employee_df['YearsExp'] = employee_df['YearsExp'].fillna(employee_df['YearsExp'].median())

employee_df = pd.get_dummies(employee_df, columns=['Degree'])
features = employee_df[['YearsExp', 'Degree_BS', 'Degree_MS', 'Degree_PhD']].values
target = employee_df['AnnualSalary'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

salary_model = LinearRegression()
salary_model.fit(X_train, y_train)

test_predictions = salary_model.predict(X_test)

print(f"Test MAE: {mean_absolute_error(y_test, test_predictions)}")
print(f"Test R2 Score: {r2_score(y_test, test_predictions)}")

print(f"Estimated salary for MS degree with 4 years experience: {salary_model.predict([[4, 0, 1, 0]])[0]}")