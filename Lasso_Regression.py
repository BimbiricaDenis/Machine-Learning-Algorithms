import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

#separating x and y
y=df['logS']
x=df.drop('logS', axis=1) 

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=100)

#Model building using lasso regression
lasso=Lasso(alpha=0.1,random_state=100)
lasso.fit(x_train, y_train)

y_lasso_training_pred=lasso.predict(x_train)
y_lasso_testing_pred=lasso.predict(x_test)

#Model evaluation
lasso_train_mse=mean_squared_error(y_train, y_lasso_training_pred)
lasso_train_r2=r2_score(y_train, y_lasso_training_pred)

lasso_test_mse=mean_squared_error(y_test, y_lasso_testing_pred)
lasso_test_r2=r2_score(y_test, y_lasso_testing_pred)

#Plotting Actual vs Predicted for Testing set
plt.figure(figsize=(5,4))
plt.scatter(x=y_test,y=y_lasso_testing_pred)
plt.title('Lasso Regression: Actual vs Predicted (Testing set)')
plt.ylabel('Predicted Points')
plt.xlabel('Actual Points')
plt.plot()
plt.show()

#Printing the results
print("Lasso Regression:")
print("Training MSE:", lasso_train_mse)
print("Training R2:", lasso_train_r2)
print("Testing MSE:", lasso_test_mse)
print("Testing R2:", lasso_test_r2)