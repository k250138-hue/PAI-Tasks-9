import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression

def gradient_descent(x_vals, y_vals, learning_rate, num_epochs):
    slope = 0
    intercept = 0
    num_samples = len(x_vals)
    loss_log = []
    
    for epoch in range(num_epochs):
        predictions = slope * x_vals + intercept
        current_mse = np.mean((y_vals - predictions) ** 2)
        loss_log.append(current_mse)
        
        if epoch == 100:
            print("MSE at epoch 100 =", round(current_mse, 3))

        grad_slope = (-2 / num_samples) * np.sum(x_vals * (y_vals - predictions))
        grad_intercept = (-2 / num_samples) * np.sum(y_vals - predictions)
        
        slope = slope - learning_rate * grad_slope
        intercept = intercept - learning_rate * grad_intercept

    plt.plot(range(num_epochs), loss_log, color='blue')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.show()

    return slope, intercept


study_hours = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
exam_scores = [32, 40, 41, 50, 54, 60, 62, 70, 78, 88]

dataset = pd.DataFrame({'StudyHours': study_hours, 'ExamScores': exam_scores})
features = dataset['StudyHours'].values
target = dataset['ExamScores'].values
learning_rates = [0.001, 0.01, 0.05]
slopes = []
intercepts = []

for idx in range(3):
    computed_m, computed_b = gradient_descent(features, target, learning_rates[idx], 1001)
    slopes.append(computed_m)
    intercepts.append(computed_b)

features_2d = dataset[['StudyHours']]
sk_model = LinearRegression()
sk_model.fit(features_2d, target)

print("Scikit-learn Slope:", sk_model.coef_)
print("Scikit-learn Intercept:", sk_model.intercept_)

for idx in range(3):
    print(f"Final slope with lr={learning_rates[idx]}: {np.sum(slopes[idx])}")
    print(f"Final intercept with lr={learning_rates[idx]}: {np.sum(intercepts[idx])}")