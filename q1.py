import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

study_hours = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
exam_scores = [32, 40, 41, 50, 54, 60, 62, 70, 78, 88]

dataset = pd.DataFrame({'StudyHours': study_hours, 'ExamScores': exam_scores})
features = dataset[['StudyHours']].values
target = dataset['ExamScores'].values
regressor = LinearRegression()

regressor.fit(features, target)

print("Slope:", regressor.coef_)
print("Intercept:", regressor.intercept_)

print(f"Predicted score for 5.5 study hours: {regressor.predict([[5.5]])[0]}")
print(f"Predicted score for 8 study hours: {regressor.predict([[8]])[0]}")

sns.regplot(
    data=dataset,
    x=dataset['StudyHours'],
    y=dataset['ExamScores'],
    scatter_kws={'color': 'blue'},
    line_kws={'color': 'red'}
)
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
plt.title("Study Hours vs Exam Scores")
plt.show()

predicted_scores = regressor.predict(features)

print(f"Mean Absolute Error (MAE): {mean_absolute_error(target, predicted_scores)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(target, predicted_scores)}")
print(f"R-squared (R2) Score: {r2_score(target, predicted_scores)}")