import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

#separating x and y
y=df['logS']
x=df.drop('logS', axis=1) 

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=100)

#Model building using random forest
rf=RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_training_pred=rf.predict(x_train)
y_rf_testing_pred=rf.predict(x_test)

#Model evaluation
rf_train_mse=mean_squared_error(y_train, y_rf_training_pred)
rf_train_r2=r2_score(y_train, y_rf_training_pred)

rf_test_mse=mean_squared_error(y_test, y_rf_testing_pred)
rf_test_r2=r2_score(y_test, y_rf_testing_pred)

#Plotting Actual vs Predicted for Testing set
plt.figure(figsize=(5,4))
plt.scatter(x=y_test,y=y_rf_testing_pred)
plt.title('Linear Regression: Actual vs Predicted (Testing set)')
plt.ylabel('Predicted Points')
plt.xlabel('Actual Points')
plt.plot()
plt.show()

#Printing the results
print("\nRandom Forest Regression:")
print("Training MSE:", rf_train_mse)
print("Training R2:", rf_train_r2)
print("Testing MSE:", rf_test_mse)
print("Testing R2:", rf_test_r2)