from sklearn.linear_model import LinearRegression,Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X,y = make_regression(n_samples = 100,
	n_features = 1,n_targets = 1,
	noise = 20)
# LinearRegression using the least squares method
reg_model = LinearRegression(fit_intercept = True,
	normalize = False,copy_X = True,n_jobs = 5
)
reg_model.fit(X,y)
# the coefficient and intercept
k0 = reg_model.coef_
b0 = reg_model.intercept_

x0 = np.linspace(-3,3)
y0 = k0*x0 + b0

# uisng other method ,for example,ridge regression
# parameters:
# alpha: regularization strength
# fit_intercept,normalize,copy_X,max_iter,tol(precision of solution)
# slover: including 'auto','svd','cholesky','lsqr','sparse_cg','sag','saga'
# random_state
# and the model attributes:
# coef_,intercept_,n_iter_
# the method is same as the LinearRegression
rdg_model = Ridge(alpha = 0.1,fit_intercept = True,
	normalize = False,copy_X=True,max_iter = 1000,
	tol = 0.01,solver = "auto",random_state = None)
rdg_model.fit(X,y)

k1 = rdg_model.coef_
b1 = rdg_model.intercept_

x1 = np.linspace(-3,3)
y1 = k1*x1 + b1

# using method of Lasso
# parameters:
# alpha,fit_intercept,normalize,precompute,
# copy_X,max_iter,tol(the precision),warm_start
# positive(the coefficient must be positive),random_state,selection
# attributes:
# coef_,sparse_coef_,intercept_,n_iter_
plt.figure()
plt.plot(x0,y0,c = "r")
plt.plot(x1,y1,c = "b")
plt.scatter(X,y,c = "g")
plt.show()
# the parameters:
# initialize parameters:fit_intercept,normalize,copy_X,n_jobs
# the reslut parameters:coef_,intercept_
# the functions:fit(train_X,train_y),get_params(),predict(X),score(test_X,test_y),set_params(**params)

