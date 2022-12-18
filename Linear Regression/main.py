import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])


#print(diabetes.keys());
#print("DESCR",diabetes.DESCR)
# diabetes_X = diabetes.data[:,np.newaxis,2]#This slicing only includes 2nd index and its features
#
# diabetes_X = diabetes.data #This slicing includes all the features of all the indexes

# diabetes_X = diabetes.load_diabetes()

diabetes_X = np.array([[1],[2],[3]])

#print(diabetes_X)
diabetes_X_train = diabetes_X
diabetes_X_test = diabetes_X

diabetes_Y_train = np.array([3,2,4])
diabetes_Y_test = np.array([3,2,4])

#Making a linear model
model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)

#Mean Squared Error
print("Mean Squared Error: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))

#Weight
print("Weight: ", model.coef_)

#Intercept
print("Intercept: ",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()

# Mean Squared Error:  3035.060115291269
# w = Weight:  [941.43097333]
# b = Intercept: 153.39713623331644