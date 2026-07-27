import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_decision_regions
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import pair_confusion_matrix
import time

def ourFunction(t1, t2, b, deg):
    counter = 0
    result = b[counter]
    counter += 1
    for i in range(1,deg+1):
        for j in range(i,0,-1):
            result += b[counter]*t1**j*t2**(i-j)
            counter += 1
    return result

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
data = pd.read_csv('male_teams.csv') # put the csv file in the same folder as the program
print("\nDimensions of the dataset:", data.shape)
print("\nColumns of the dataset \"male_teams.csv\":")
print(data.columns)
print("\nStatistics of the \"overall\" column:")
print(data["overall"].describe())
Matr_cor=data.corr(numeric_only=True)
print("\nCorrelation matrix for the \"overall\" column:")
print(Matr_cor["overall"].sort_values(ascending=False))
print("\nNumber of null/missing values of the column \"overall\":", data["overall"].isnull().sum())
print("Number of null/missing values of the column \"midfield\":", data["midfield"].isnull().sum())
print("Number of null/missing values of the column \"defence\":", data["defence"].isnull().sum())
print("As we can see, these 3 columns don't have null or missing values, so we will use them for regression: we will predict the column \"overall\" with either the column \"midfield\" or the columns \"midfield\" and \"defence\".")
print("\nWe'll add a column named Tier, with the following values: 1 if their value in the column \"overall\" is at least 69, which is the median value of the column, and 0 otherwise. We will classify the teams based on their value in the column \"Tier\".")
ts = data.values[:, 11].reshape((-1,1)) #overall = 11
tier = []
for z in ts:
    if (z >= 69):
        tier.append(1)
    else: tier.append(0)
data['Tier'] = tier
print("\nWe will use only the numerical columns.")
data = data.select_dtypes(include=np.number)
print("\nDimensions of the new table with only numerical columns:", data.shape)
print("\nColumns of the new table with only numerical columns:")
print(data.columns)
print("\nNumber of null/missing values of the new table:")
print(data.isnull().sum())
print("\nWe will use an imputer to fill the NaN values of the numerical columns with the mean value of each column respectively.")
x_class = data[data.columns[:-1]]
imputer = imputer.fit(x_class)
x_class = imputer.transform(x_class)
y_class = data.Tier
x_class, y_class = np.array(x_class), np.array(y_class)


#Univariate Linear Regression
print("\nUnivariate Linear Regression")
x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
timer = time.time()
model = LinearRegression().fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
plt.plot(x_test,y_test,'ro')
plt.plot(x_test,y_predict)
plt.xlabel('Midfield')
plt.ylabel('Overall')
plt.title("Univariate Linear Regression")
plt.show()

#Univariate Polynomial Regression
for i in range(2,9):
    print("\nUnivariate Polynomial Regression of Degree",i)
    x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
    x = PolynomialFeatures(degree=i, include_bias=False).fit_transform(x)
    y = data.values[:, 6]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    timer = time.time()
    model = LinearRegression().fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
    plt.figure(dpi=100)
    plt.plot(x_test[:,i-1],y_test,'ro')
    y_pred_indices = np.argsort(x_test[:,i-1])
    y_predict = np.take_along_axis(y_predict, y_pred_indices, axis=0)
    x_test[:,i-1] = np.sort(x_test[:,i-1])
    plt.plot(x_test[:,i-1],y_predict)
    plt.xlabel('Midfield')
    plt.ylabel('Overall')
    plt.title("Univariate Polynomial Regression of Degree "+str(i))
    plt.show()

#Bivariate Linear Regression
print("\nBivariate Linear Regression")
x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
x = np.append(x, data.values[:, 9].reshape((-1,1)), axis=1)
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
timer = time.time()
model = LinearRegression().fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
ax1 = plt.axes(projection = '3d')
x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
X1, X2 = np.meshgrid(x1, x2)
Y = ourFunction(X1, X2, y_predict, 1)
ax1.plot_surface(X1, X2, Y, alpha=0.6)
plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
plt.title("Bivariate Linear Regression")
plt.show()

#Bivariate Polynomial Regression
for i in range(2,9):
    print("\nBivariate Polynomial Regression of Degree",i)
    x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
    x = np.append(x, data.values[:, 9].reshape((-1,1)), axis=1)
    x = PolynomialFeatures(degree=i, include_bias=False).fit_transform(x)
    y = data.values[:, 6]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    timer = time.time()
    model = LinearRegression().fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
    plt.figure(dpi=100)
    ax1 = plt.axes(projection = '3d')
    x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
    x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Y = ourFunction(X1, X2, y_test, i)
    ax1.plot_surface(X1, X2, Y, alpha=0.6)
    plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
    plt.title("Bivariate Polynomial Regression of Degree "+str(i))
    plt.show()


x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#Univariate Ridge Regression
print("\nUnivariate Ridge Regression")
timer = time.time()
model = Ridge(random_state=0).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
plt.plot(x_test,y_test,'ro')
plt.plot(x_test,y_predict)
plt.xlabel('Midfield')
plt.ylabel('Overall')
plt.title("Univariate Ridge Regression")
plt.show()

#Univariate Lasso Regression
print("\nUnivariate Lasso Regression")
timer = time.time()
model = Lasso(random_state=0).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
plt.plot(x_test,y_test,'ro')
plt.plot(x_test,y_predict)
plt.xlabel('Midfield')
plt.ylabel('Overall')
plt.title("Univariate Lasso Regression")
plt.show()

#Univariate ElasticNet Regression
print("\nUnivariate ElasticNet Regression")
timer = time.time()
model = ElasticNet(random_state=0).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
plt.plot(x_test,y_test,'ro')
plt.plot(x_test,y_predict)
plt.xlabel('Midfield')
plt.ylabel('Overall')
plt.title("Univariate ElasticNet Regression")
plt.show()

x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
x = np.append(x, data.values[:, 9].reshape((-1,1)), axis=1)
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#Bivariate Ridge Regression
print("\nBivariate Ridge Regression")
timer = time.time()
model = Ridge(random_state=0).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
ax1 = plt.axes(projection = '3d')
x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
X1, X2 = np.meshgrid(x1, x2)
Y = ourFunction(X1, X2, y_predict, 1)
ax1.plot_surface(X1, X2, Y, alpha=0.6)
plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
plt.title("Bivariate Ridge Regression")
plt.show()

#Bivariate Lasso Regression
print("\nBivariate Lasso Regression")
timer = time.time()
model = Lasso(random_state=0).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
ax1 = plt.axes(projection = '3d')
x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
X1, X2 = np.meshgrid(x1, x2)
Y = ourFunction(X1, X2, y_predict, 1)
ax1.plot_surface(X1, X2, Y, alpha=0.6)
plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
plt.title("Bivariate Lasso Regression")
plt.show()

#Bivariate ElasticNet Regression
print("\nBivariate ElasticNet Regression")
timer = time.time()
model = ElasticNet(random_state=0).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
ax1 = plt.axes(projection = '3d')
x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
X1, X2 = np.meshgrid(x1, x2)
Y = ourFunction(X1, X2, y_predict, 1)
ax1.plot_surface(X1, X2, Y, alpha=0.6)
plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
plt.title("Bivariate ElasticNet Regression")
plt.show()


#Logistic Regression
print("\nLogistic Regression")
x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
timer = time.time()
model = LogisticRegression(random_state=0).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
print('Classification report:', classification_report(y_test, y_predict), sep='\n')
plt.figure(dpi=100)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
plt.show()
plt.figure(dpi=100)
plot_decision_regions(X = x_test[:,[8,9]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[8,9]],y_train))
plt.title("Logistic Regression")
plt.show()


#K-NN Classification
for i in range(1,10,2):
    print("K-NN with k =",i)
    x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
    timer = time.time()
    model = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    mcc = matthews_corrcoef(y_test, y_predict)
    print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
    print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
    print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
    print('Classification report:', classification_report(y_test, y_predict), sep='\n')
    plt.figure(dpi=100)
    plt.title("K-NN with k = "+str(i)+" Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
    plt.show()
    plt.figure(dpi=100)
    plot_decision_regions(X = x_test[:,[6,8]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[6,8]],y_train))
    plt.title("K-NN with k = "+str(i))
    plt.show()


#Linear SVC
print("Linear SVC")
x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
timer = time.time()
model = Pipeline([("linear_svc", LinearSVC(C=2, random_state=0))]).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
mcc = matthews_corrcoef(y_test, y_predict)
print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
print('Classification report:', classification_report(y_test, y_predict), sep='\n')
plt.figure(dpi=100)
plt.title("Linear SVC Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
plt.show()
plt.figure(dpi=100)
plot_decision_regions(X = x_test[:,[8,9]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[8,9]],y_train))
plt.title("Linear SVC")
plt.show()

#Polynomial SVC
print("Polynomial SVC of Degree 2")
x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
timer = time.time()
model = Pipeline([("svm_clf", SVC(kernel="poly", degree=2, C=2, random_state=0))]).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
mcc = matthews_corrcoef(y_test, y_predict)
print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
print('Classification report:', classification_report(y_test, y_predict), sep='\n')
plt.figure(dpi=100)
plt.title("Polynomial SVC of Degree 2 Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
plt.show()
plt.figure(dpi=100)
plot_decision_regions(X = x_test[:,[8,9]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[8,9]],y_train))
plt.title("Polynomial SVC of Degree 2")
plt.show()

#RBF SVC
print("RBF SVC of Degree 2")
x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
timer = time.time()
model = Pipeline([("svm_clf", SVC(kernel="rbf", degree=2, C=2, random_state=0))]).fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
mcc = matthews_corrcoef(y_test, y_predict)
print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
print('Classification report:', classification_report(y_test, y_predict), sep='\n')
plt.figure(dpi=100)
plt.title("RBF SVC of Degree 2 Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
plt.show()
plt.figure(dpi=100)
plot_decision_regions(X = x_test[:,[8,9]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[8,9]],y_train))
plt.title("RBF SVC of Degree 2")
plt.show()


#Univariate Linear SVR
print("Univariate Linear SVR")
x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
timer = time.time()
model = LinearSVR(C=2, random_state=0).fit(x_train,y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
plt.plot(x_test,y_test,'ro')
plt.plot(x_test,y_predict)
plt.xlabel('Midfield')
plt.ylabel('Overall')
plt.title("Univariate Linear SVR")
plt.show()

#Univariate Polynomial SVR
print("\n\nUnivariate Polynomial SVR of Degree 2")
x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
timer = time.time()
model = SVR(kernel="poly", degree=2, C=1).fit(x_train,y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
plt.plot(x_test,y_test,'ro')
y_pred_indices = np.argsort(x_test[:,0])
y_predict = np.take_along_axis(y_predict, y_pred_indices, axis=0)
x_test[:,0] = np.sort(x_test[:,0])
plt.plot(x_test[:,0],y_predict)
plt.xlabel('Midfield')
plt.ylabel('Overall')
plt.title("Univariate Polynomial SVR of Degree 2")
plt.show()

#Bivariate Linear SVR
print("\n\nBivariate Linear SVR")
x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
x = np.append(x, data.values[:, 9].reshape((-1,1)), axis=1)
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
timer = time.time()
model = LinearSVR(C=2, random_state=0).fit(x_train,y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
ax1 = plt.axes(projection = '3d')
x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
X1, X2 = np.meshgrid(x1, x2)
Y = ourFunction(X1, X2, y_predict, 1)
ax1.plot_surface(X1, X2, Y, alpha=0.6)
plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
plt.title("Bivariate Linear SVR")
plt.show()

#Bivariate Polynomial SVR
print("\n\nBivariate Polynomial SVR of Degree 2")
x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
x = np.append(x, data.values[:, 9].reshape((-1,1)), axis=1)
y = data.values[:, 6]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
timer = time.time()
model = SVR(kernel="poly", degree=2, C=1).fit(x_train,y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
plt.figure(dpi=100)
ax1 = plt.axes(projection = '3d')
x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
X1, X2 = np.meshgrid(x1, x2)
Y = ourFunction(X1, X2, y_predict, 1)
ax1.plot_surface(X1, X2, Y, alpha=0.6)
plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
plt.title("Bivariate Polynomial SVR of Degree 2")
plt.show()


#Univariate Decision Tree Regression
for i in range(1,6):
    print("\nUnivariate Decision Tree Regression of Depth",i)
    x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
    y = data.values[:, 6]
    x, y = np.array(x), np.array(y)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    timer = time.time()
    model = DecisionTreeRegressor(max_depth=i, random_state=0).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
    plt.figure(dpi=100)
    tree.plot_tree(model, filled = True)
    plt.show()
    plt.figure(dpi=100)
    plt.plot(x_test,y_test,'ro')
    y_pred_indices = np.argsort(x_test[:,0])
    y_predict = np.take_along_axis(y_predict, y_pred_indices, axis=0)
    x_test = np.sort(x_test[:,0])
    plt.plot(x_test,y_predict)
    plt.xlabel('Midfield')
    plt.ylabel('Overall')
    plt.title("Univariate Decision Tree Regression of Depth "+str(i))
    plt.show()

#Bivariate Decision Tree Regression
for i in range(1,6):
    print("\nBivariate Decision Tree Regression of Depth",i)
    x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
    x = np.append(x, data.values[:, 9].reshape((-1,1)), axis=1)
    y = data.values[:, 6]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    timer = time.time()
    model = DecisionTreeRegressor(max_depth=i, random_state=0).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
    plt.figure(dpi=100)
    tree.plot_tree(model, filled = True)
    plt.show()
    plt.figure(dpi=100)
    ax1 = plt.axes(projection = '3d')
    x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
    x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Y = ourFunction(X1, X2, y_test, i)
    ax1.plot_surface(X1, X2, Y, alpha=0.6)
    plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
    plt.title("Bivariate Decision Tree Regression of Depth "+str(i))
    plt.show()


#Univariate Random Forest Regression
for i in range(1,6):
    print("\nUnivariate Random Forest Regression of Depth",i)
    x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
    y = data.values[:, 6]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    timer = time.time()
    model = RandomForestRegressor(max_depth=i, random_state=0).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
    plt.figure(dpi=100)
    plt.plot(x_test,y_test,'ro')
    y_pred_indices = np.argsort(x_test[:,0])
    y_predict = np.take_along_axis(y_predict, y_pred_indices, axis=0)
    x_test = np.sort(x_test[:,0])
    plt.plot(x_test,y_predict)
    plt.xlabel('Midfield')
    plt.ylabel('Overall')
    plt.title("Univariate Random Forest Regression of Depth "+str(i))
    plt.show()

#Bivariate Random Forest Regression
for i in range(1,6):
    print("\nBivariate Random Forest Regression of Depth",i)
    x = data.values[:, 8].reshape((-1,1)) #overall=6, midfield=8 ,defence=9
    x = np.append(x, data.values[:, 9].reshape((-1,1)), axis=1)
    y = data.values[:, 6]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    timer = time.time()
    model = RandomForestRegressor(max_depth=i, random_state=0).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Coefficient R^2:', model.score(x_test, y_test)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict)) #MAE
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict)) #MSE
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict))) #RMSE
    plt.figure(dpi=100)
    ax1 = plt.axes(projection = '3d')
    x1 = np.linspace(min(x_test[:,0]), max(x_test[:,0]), 1000)
    x2 = np.linspace(min(x_test[:,1]), max(x_test[:,1]), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Y = ourFunction(X1, X2, y_test, i)
    ax1.plot_surface(X1, X2, Y, alpha=0.6)
    plt.scatter(x_test[:,0].astype(np.float64), x_test[:,1].astype(np.float64), y_test.astype(np.float64), 'red')
    plt.title("Bivariate Random Forest Regression of Depth "+str(i))
    plt.show()


#Decision Tree Classification
for i in range(1,6):
    print("\nDecision Tree Classification of Depth",i)
    x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
    timer = time.time()
    model = DecisionTreeClassifier(max_depth=i, random_state=0).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    mcc = matthews_corrcoef(y_test, y_predict)
    print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
    print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
    print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
    print('Classification report:', classification_report(y_test, y_predict), sep='\n')
    plt.figure(dpi=100)
    tree.plot_tree(model, filled = True)
    plt.show()
    plt.figure(dpi=100)
    plt.title("Decision Tree Classification of Depth "+str(i)+" Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
    plt.show()
    plt.figure(dpi=100)
    plot_decision_regions(X = x_test[:,[6,8]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[6,8]],y_train))
    plt.title("Decision Tree Classification of Depth "+str(i))
    plt.show()


#Random Forest Classification
for i in range(1,6):
    print("\nRandom Forest Classification of Depth",i)
    x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
    timer = time.time()
    model = RandomForestClassifier(max_depth=i, random_state=0).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    mcc = matthews_corrcoef(y_test, y_predict)
    print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
    print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
    print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
    print('Classification report:', classification_report(y_test, y_predict), sep='\n')
    plt.figure(dpi=100)
    plt.title("Random Forest Classification of Depth "+str(i)+" Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
    plt.show()
    plt.figure(dpi=100)
    plot_decision_regions(X = x_test[:,[6,8]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[6,8]],y_train))
    plt.title("Random Forest Classification of Depth "+str(i))
    plt.show()


#Gaussian NB
print("\nGaussian Naive Bayes")
x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
timer = time.time()
model = GaussianNB().fit(x_train, y_train)
y_predict = model.predict(x_test)
timer = time.time() - timer
mcc = matthews_corrcoef(y_test, y_predict)
print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
print('Total time:', timer,'seconds')
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
print('Classification report:', classification_report(y_test, y_predict), sep='\n')
plt.figure(dpi=100)
plt.title("Gaussian Naive Bayes Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
plt.show()
plt.figure(dpi=100)
plot_decision_regions(X = x_test[:,[8,9]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[8,9]],y_train))
plt.title("Gaussian Naive Bayes")
plt.show()


#PCA Logistic Regression
for i in range(1,11):
    if i==1:
        print("\nLogistic Regression with PCA of",i,"component")
    else:
        print("\n\nLogistic Regression with PCA of",i,"components")
    x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    timer = time.time()
    model = LogisticRegression(random_state=0).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
    print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
    print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
    print('Classification report:', classification_report(y_test, y_predict), sep='\n')
    plt.figure(dpi=100)
    if i==1:
        plt.title("Logistic Regression with PCA of "+str(i)+" component Confusion Matrix")
    else:
        plt.title("Logistic Regression with PCA of "+str(i)+" components Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
    plt.show()
    plt.figure(dpi=100)
    if i==1:
        plot_decision_regions(X = x_test, y = y_test.astype(np.int_), clf = model.fit(x_train,y_train))
        plt.title("Logistic Regression with PCA of "+str(i)+" component")
    else:
        plot_decision_regions(X = x_test[:,[0,1]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[0,1]],y_train))
        plt.title("Logistic Regression with PCA of "+str(i)+" components")
    plt.show()


#PCA Linear SVC
for i in range(1,11):
    if i==1:
        print("\nLinear SVC with PCA of",i,"component")
    else:
        print("\nLinear SVC with PCA of",i,"components")
    x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    timer = time.time()
    model = Pipeline([("linear_svc", LinearSVC(C=2, random_state=0))]).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
    print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
    print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
    print('Classification report:', classification_report(y_test, y_predict), sep='\n')
    plt.figure(dpi=100)
    if i==1: 
        plt.title("Linear SVC with PCA of "+str(i)+" component Confusion Matrix")
    else:
        plt.title("Linear SVC with PCA of "+str(i)+" components Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
    plt.show()
    plt.figure(dpi=100)
    if i==1:
        plot_decision_regions(X = x_test, y = y_test.astype(np.int_), clf = model.fit(x_train,y_train))
        plt.title("Linear SVC with PCA of "+str(i)+" component")
    else:
        plot_decision_regions(X = x_test[:,[0,1]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[0,1]],y_train))
        plt.title("Linear SVC with PCA of "+str(i)+" components")
    plt.show()


#PCA Gaussian NB
for i in range(1,11):
    if i==1:
        print("\nGaussian Naive Bayes with PCA of",i,"component")
    else:
        print("\nGaussian Naive Bayes with PCA of",i,"components")
    x_train,x_test,y_train,y_test = train_test_split(x_class,y_class,test_size=0.3,random_state=0)
    pca = PCA(n_components=i)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    timer = time.time()
    model = GaussianNB().fit(x_train, y_train)
    y_predict = model.predict(x_test)
    timer = time.time() - timer
    print('Accuracy score:', accuracy_score(y_test, y_predict)*100, '%')
    print('Matthew correlation coefficient:', matthews_corrcoef(y_test, y_predict)*100, '%')
    print('Total time:', timer,'seconds')
    print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
    print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
    print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
    print('Confusion matrix:', confusion_matrix(y_test, y_predict), sep='\n')
    print('Classification report:', classification_report(y_test, y_predict), sep='\n')
    plt.figure(dpi=100)
    if i==1:
        plt.title("Gaussian NB with PCA of "+str(i)+" component Confusion Matrix")
    else:
        plt.title("Gaussian NB with PCA of "+str(i)+" components Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues')
    plt.show()
    plt.figure(dpi=100)
    if i==1:
        plot_decision_regions(X = x_test, y = y_test.astype(np.int_), clf = model.fit(x_train,y_train))
        plt.title("Gaussian Naive Bayes with PCA of "+str(i)+" component")
    else:
        plot_decision_regions(X = x_test[:,[0,1]], y = y_test.astype(np.int_), clf = model.fit(x_train[:,[0,1]],y_train))
        plt.title("Gaussian Naive Bayes with PCA of "+str(i)+" components")
    plt.show()


#PCA Data Visualization
print("\nPCA Data Visualization with 1 component")
pca = PCA(n_components=1)
x_pca = pca.fit_transform(x_class)
print('Explained variability per principal component: {}'.format(pca.explained_variance_ratio_))
plt.figure()
plt.scatter(x_pca, x_pca, c=y_class.astype(np.int_))
plt.title("PCA Data Visualization with 1 component")
plt.show()

print("\nPCA Data Visualization with 2 components")
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_class)
print('Explained variability per principal component: {}'.format(pca.explained_variance_ratio_))
plt.figure()
plt.scatter(x_pca[:,0], x_pca[:,1], c=y_class.astype(np.int_))
plt.title("PCA Data Visualization with 2 components")
plt.show()

print("\nPCA Data Visualization with 3 components")
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_class)
print('Explained variability per principal component: {}'.format(pca.explained_variance_ratio_))
plt.figure()
ax1 = plt.axes(projection = '3d')
ax1.scatter(x_pca[:,0], x_pca[:,1], x_pca[:,2], c=y_class.astype(np.int_))
plt.title("PCA Data Visualization with 3 components")
plt.show()


#t-SNE Data Visualization
print("\nt-SNE Data Visualization with 1 component")
tsne = TSNE(n_components=1)
x_tsne = tsne.fit_transform(x_class)
plt.figure()
plt.scatter(x_tsne, x_tsne, c=y_class.astype(np.int_))
plt.title("t-SNE Data Visualization with 1 component")
plt.show()

print("\nt-SNE Data Visualization with 2 components")
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(x_class)
plt.figure()
plt.scatter(x_tsne[:,0], x_tsne[:,1], c=y_class.astype(np.int_))
plt.title("t-SNE Data Visualization with 2 components")
plt.show()

print("\nt-SNE Data Visualization with 3 components")
tsne = TSNE(n_components=3)
x_tsne = tsne.fit_transform(x_class)
plt.figure()
ax1 = plt.axes(projection = '3d')
ax1.scatter(x_tsne[:,0], x_tsne[:,1], x_tsne[:,2], c=y_class.astype(np.int_))
plt.title("t-SNE Data Visualization with 3 components")
plt.show()


#K-Means Clustering
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
print("\nK-Means Clustering")
for i in range(2,11):
    print("\nK-Means Clustering with",i,"clusters")
    timer = time.time()
    model = KMeans(n_clusters=i, n_init='auto', random_state=0).fit(x_class)
    timer = time.time() - timer
    print("The Pair Confusion Matrix is:")
    print(pair_confusion_matrix(x_class[:,6], model.labels_))
    print("The Silhouette score is:",silhouette_score(x_class, model.labels_))
    print("The Davies Bouldin score is:",davies_bouldin_score(x_class, model.labels_))
    print("The Calinski Harabasz score is:",calinski_harabasz_score(x_class, model.labels_))
    print("Total time:", timer,"seconds")
    centroids = model.cluster_centers_
    plt.figure()
    plt.scatter(x_class[:,6], x_class[:,8], c=model.labels_.astype(float))
    plt.scatter(centroids[:,6], centroids[:,8], c='black')
    plt.title("2D K-Means Clustering with "+str(i)+" clusters")
    plt.show()
    plt.figure()
    ax1 = plt.axes(projection = '3d')
    ax1.scatter(x_class[:,6], x_class[:,8], x_class[:,9], c=model.labels_.astype(float))
    ax1.scatter(centroids[:,6], centroids[:,8], centroids[:, 9], c='black')
    plt.title("3D K-Means Clustering with "+str(i)+" clusters")
    plt.show()
    distortions.append(sum(np.min(cdist(x_class, model.cluster_centers_, 'euclidean'), axis=1)**2) / x_class.shape[0])
    inertias.append(model.inertia_)
    mapping1[i] = distortions[-1]
    mapping2[i] = inertias[-1]

print("\nK-Means distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')
plt.plot(range(2,11), distortions)
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('The K-Means Elbow method using Distortion')
plt.grid()
plt.show()

print("\nK-Means inertia values:")
for key, val in mapping2.items():
    print(f'{key} : {val}')
plt.plot(range(2,11), inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The K-Means Elbow method using Inertia')
plt.grid()
plt.show()


#DBSCAN Clustering
print("\nDBSCAN Clustering")
for i in range(1,11):
    print("\nDBSCAN Clustering with EPS",i)
    timer = time.time()
    model = DBSCAN(eps=i).fit(x_class)
    timer = time.time() - timer
    print("The Pair Confusion Matrix is:")
    print(pair_confusion_matrix(x_class[:,6], model.labels_))
    print("The Silhouette score is:",silhouette_score(x_class, model.labels_))
    print("The Davies Bouldin score is:",davies_bouldin_score(x_class, model.labels_))
    print("The Calinski Harabasz score is:",calinski_harabasz_score(x_class, model.labels_))
    print("Total time:", timer,"seconds")
    plt.figure()
    plt.scatter(x_class[:,6], x_class[:,8], c=model.labels_.astype(float))
    plt.title("2D DBSCAN Clustering with EPS "+str(i))
    plt.show()
    plt.figure()
    ax1 = plt.axes(projection = '3d')
    ax1.scatter(x_class[:,6], x_class[:,8], x_class[:,9], c=model.labels_.astype(float))
    plt.title("3D DBSCAN Clustering with EPS "+str(i))
    plt.show()
