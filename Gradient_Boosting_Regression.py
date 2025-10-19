import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt 

#importing the dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

#separating x and y
y=df['logS']
x=df.drop('logS', axis=1) 

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=100)

#Model building using gradient boosting regression
gbr=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=100)
gbr.fit(x_train, y_train)

y_gbr_training_pred=gbr.predict(x_train)
y_gbr_testing_pred=gbr.predict(x_test)

#Model evaluation
gbr_train_mse=mean_squared_error(y_train, y_gbr_training_pred)
gbr_train_r2=r2_score(y_train, y_gbr_training_pred)

gbr_test_mse=mean_squared_error(y_test, y_gbr_testing_pred)
gbr_test_r2=r2_score(y_test, y_gbr_testing_pred)

#Plotting Actual vs Predicted for Testing set
plt.figure(figsize=(6,4))
plt.scatter(x=y_test,y=y_gbr_testing_pred)
plt.title('Gradient Boosting Regression: Actual vs Predicted (Testing set)')
plt.ylabel('Predicted Points')
plt.xlabel('Actual Points')
plt.plot()
plt.show()

#Printing the results
print("Gradient Boosting Regression:")
print("Training MSE:", gbr_train_mse)
print("Training R2:", gbr_train_r2)
print("Testing MSE:", gbr_test_mse)
print("Testing R2:", gbr_test_r2)