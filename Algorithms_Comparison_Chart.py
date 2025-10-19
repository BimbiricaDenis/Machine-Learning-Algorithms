import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt 

#importing the dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

#separating x and y
y=df['logS']
x=df.drop('logS', axis=1) 

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=100)

#Model building using linear regression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_lr_training_pred=lr.predict(x_train)
y_lr_testing_pred=lr.predict(x_test)

#Model building using random forest
rf=RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_training_pred=rf.predict(x_train)
y_rf_testing_pred=rf.predict(x_test)

#Model biulding using ridge regression
ridge=Ridge(alpha=1.0,random_state=100)
ridge.fit(x_train, y_train)

y_ridge_training_pred=ridge.predict(x_train)
y_ridge_testing_pred=ridge.predict(x_test)

#Model building using lasso regression
lasso=Lasso(alpha=0.1,random_state=100)
lasso.fit(x_train, y_train)

y_lasso_training_pred=lasso.predict(x_train)
y_lasso_testing_pred=lasso.predict(x_test)

#Model building using gradient boosting regression
gbr=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=100)
gbr.fit(x_train, y_train)

y_gbr_training_pred=gbr.predict(x_train)
y_gbr_testing_pred=gbr.predict(x_test)

#Model evaluation
lr_train_mse=mean_squared_error(y_train, y_lr_training_pred)
lr_train_r2=r2_score(y_train, y_lr_training_pred)

lr_test_mse=mean_squared_error(y_test, y_lr_testing_pred)
lr_test_r2=r2_score(y_test, y_lr_testing_pred)

rf_train_mse=mean_squared_error(y_train, y_rf_training_pred)
rf_train_r2=r2_score(y_train, y_rf_training_pred)

rf_test_mse=mean_squared_error(y_test, y_rf_testing_pred)
rf_test_r2=r2_score(y_test, y_rf_testing_pred)

ridge_train_mse=mean_squared_error(y_train, y_ridge_training_pred)
ridge_train_r2=r2_score(y_train, y_ridge_training_pred)

ridge_test_mse=mean_squared_error(y_test, y_ridge_testing_pred)
ridge_test_r2=r2_score(y_test, y_ridge_testing_pred)

lasso_train_mse=mean_squared_error(y_train, y_lasso_training_pred)
lasso_train_r2=r2_score(y_train, y_lasso_training_pred)

lasso_test_mse=mean_squared_error(y_test, y_lasso_testing_pred)
lasso_test_r2=r2_score(y_test, y_lasso_testing_pred)

gbr_train_mse=mean_squared_error(y_train, y_gbr_training_pred)
gbr_train_r2=r2_score(y_train, y_gbr_training_pred)

gbr_test_mse=mean_squared_error(y_test, y_gbr_testing_pred)
gbr_test_r2=r2_score(y_test, y_gbr_testing_pred)

#Plotting Model Comparison based on R2 Score
algorithms=['Linear Regression', 'Random Forest', 'Ridge Regression', 'Lasso Regression', 'Gradient Boosting']
test_r2_scores=[lr_test_r2, rf_test_r2, ridge_test_r2, lasso_test_r2, gbr_test_r2]
plt.figure(figsize=(10,5))
plt.bar(algorithms, test_r2_scores)
plt.ylabel('R2 Score on Testing Sets')
plt.title('Model Comparison based on R2 Score')
plt.show()


