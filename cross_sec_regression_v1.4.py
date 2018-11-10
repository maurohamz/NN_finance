#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:29:02 2018

@author: maurohamz
"""
import random
import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.api import add_constant
from scipy import stats

###############################################################################

"""
Data loading and preparation
"""

# Load data from SP500

df1 = pd.read_csv('/Users/maurohamz/Desktop/TA/RA/DNN/sp200_data_.csv', header = 0 )
df1.head()

n_stocks = df1.Name.nunique()
n_factors = len(df1.columns) - len(['Date', 'Name', 'Px_LAST', 'CUR_MKT_CAP', 'Return']) # List of non-factor columns in data

print "Number of stocks:",n_stocks
print "Number of stocks:",n_factors

# Benchmark and Risk-free Rate data

bench = pd.read_csv('/Users/maurohamz/Desktop/TA/RA/DNN/sp500_ret.csv', header = 0 )
bench.head()
rf = pd.read_csv('/Users/maurohamz/Desktop/TA/RA/DNN/risk_free_ret.csv', header = 0 )
rf.head() 

# Clean Data

df1 = df1.apply(pd.to_numeric, errors='coerce')
df1 = df1.fillna( 0.0 )
df1 = df1.replace('#VALUE!',0.0)
df1 = df1.replace('#DIV/0!',0.0)
df1 = df1.replace('#N/A',0.0)
df1 = df1.replace('#N/A N/A',0.0)
df1 = df1.replace('#N/A N/A ',0.0) 
df1 = df1.replace(' ',0.0)                   

# Split data for training and testing

test_size = 0.20
n_test_obs = int(math.ceil(len(df1.index)*test_size)) # this rounds up

X = np.delete( df1.values, [0,1,2,3,10] , axis = 1 )
y = np.delete( df1.values, [0,1,2,3,4,5,6,7,8,9], axis= 1 )

X_train_orig, X_test_orig, y_train, y_test = model_selection.train_test_split( X, y.ravel(), test_size = 0.20, shuffle = False )

train_periods = int(len(df1.index)/n_stocks*(1-test_size))  # this rounds down
#train_periods = 3

# Fix random seed for reproducibility

seed = 7
np.random.seed(seed)

# Normalizing Data:

# Using sklearn normalizer

#norm = preprocessing.Normalizer().fit(X_train_orig)
#X_train = norm.transform(X_train_orig)
#X_test = norm.transform(X_test_orig)

# Regular normalizer: calculate mean and std from train to normalize test

x_mean = []
x_std = []

for i in range(0,n_factors):
    x_mean.append(np.mean(X_train_orig[:,i]))
    x_std.append(np.std(X_train_orig[:,i]))
    
X_train = X_train_orig
X_test = X_test_orig    

for j in range(0,n_factors):
    X_train[:,j] = (X_train_orig[:,j] - x_mean[j])/x_std[j]
    X_test[:,j] = (X_test_orig[:,j] - x_mean[j])/x_std[j]
    
"""
  Setting up predictive model, NN and Linear Regression. Choose the one to use and comment the other one
"""

# Sklearn LinearRegression:

lm = LinearRegression(fit_intercept = False)        # this line defines a linear predictive model

# NN using sklearn wrapper

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='linear'))
	model.add(Dense(1, kernel_initializer='normal')) # output layer
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
	return model

def NN1_model(n_hidden_neurons, l1_reg):
        model = Sequential()
        model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
        model.add(Dense(n_hidden_neurons, input_dim=6, kernel_initializer='normal',kernel_regularizer=l1(l1_reg), activation='relu'))
    	model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
        return model


#lm = KerasRegressor(build_fn=NN1_model, epochs=50, batch_size=1, verbose=1)  # this line defines a NN predictive model

# NN parameter tuning:

def parameter_tuning(X, y, cv=3, seed = 7):

    
  early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')    

  param_grid = dict(n_hidden_neurons=[10,25,50,100], l1_reg = [0.05, 0.1, 0.2]) # dropout=[0, 0.1, 0.2, 0.3]
  estimator = KerasRegressor(build_fn=NN1_model, epochs=100, batch_size=10, verbose=1)
  grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,fit_params=dict(callbacks=[early_stopping]))
  grid_result = grid.fit(X, y)

  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
# Result of tuning -> mean_squared_error: 0.0068; Best: -0.008039 using {'l1_reg': 0.05, 'n_hidden_neurons': 10}

# Testing with a low number of period to compare errors between NN and Linear Regression

#lm.fit(X_train[0:218], y_train[0:218])
#predictions_lm = lm.predict( X_train[218:(218+n_stocks)] )
#predictions_insample_ = lm.predict (X_train[0:218])
        
lm.fit(X_train, y_train)
predictions_lm = lm.predict( X_test )
predictions_insample_ = lm.predict (X_train)
        

"""
Goodnes-of-fit Statistics
"""
plt.plot(y_train)
plt.ylabel('Stock Returns', fontsize = 14)
plt.show()

factors_ = df1[['CURR_ENTP_VAL', 'PX_TO_BOOK_RATIO', 'CURRENT_EV_TO_T12M_EBITDA', 'PX_TO_SALES_RATIO', 'PE_RATIO', 'Log_CAP']].copy()
print "Factor Correlation Matrix:"
plt.matshow(factors_.corr())
plt.colorbar()
plt.show() 

print ("\nModel Statistics:\n")
#print 'Test period MSE out of sample:', mean_squared_error(y_train[218:(218+n_stocks)],predictions_lm)
#print 'Test Period MAE out of sample:', mean_absolute_error(y_train[218:(218+n_stocks)],predictions_lm)
#print 'Test period MSE in-sample:', mean_squared_error(y_train[0:218], predictions_insample_)

print 'Test period MSE out of sample:', mean_squared_error(y_test,predictions_lm)
print 'Test period MAE out of sample:', mean_absolute_error(y_test,predictions_lm)
print 'Test period MSE in-sample:', mean_squared_error(y_train, predictions_insample_)
print '\nR^2:', r2_score(y_test,predictions_lm)
print 'explained variance:', explained_variance_score(y_test,predictions_lm)
                    
# Template for using statsmodel OLS

#ols_results = sm.OLS(y_train, X_train).fit()
#print ols_results.summary()
#ols_results.pvalues[0:5]

"""
 Cross-sectional regression over time
"""

mse_cs = []
mse_insample = []
predictions_cs = []
predictions_insample = []
predictions_insample_set = []
coefficients = []
r2 = []
slice_ = []
prediction_set = []
p_values = []

for k in range(0,n_stocks*train_periods,n_stocks):
    lm.fit(X_train[k:k+n_stocks,:],y_train[k:(k+n_stocks)])
    
    #slice_1 = len(X_train[k:k+218])
    #print 'slice1:', slice_1
    predictions_cs = lm.predict(X_train[(k+n_stocks):(k+n_stocks*2),:])  
    prediction_set.append(predictions_cs)
   
    ols_results = sm.OLS(y_train[k:(k+n_stocks)], X_train[k:k+n_stocks,:]).fit()
    p_values.append(ols_results.pvalues[0:5])

    #slice_2 = len(X_train[k+218:k+436])
    #print 'slice 2:', slice_2
    predictions_insample = lm.predict(X_train[k:(k+n_stocks),:])
    predictions_insample_set.append(predictions_insample) 
    
    mse_cs.append(mean_squared_error(y_train[(k+n_stocks):(k+n_stocks*2)],predictions_cs))
    mse_insample.append(mean_squared_error(y_train[k:(k+n_stocks)],predictions_insample))
    #coefficients.append(lm.coef_) # comment this if using NN 

"""
Cross-Section Diagnostics
"""

#print "Coefficients for first period:", coefficients[1] # comment this if using NN
print "\nP-values for first period:\n", p_values[1]
print "\nMSE in-sample:", np.mean(mse_insample)
print "MSE out-of-sample:\n", np.mean(mse_cs)
plt.plot(mse_cs, color = 'red', label = 'MSE')
plt.plot(mse_insample, color = 'black', label = "MSE In-Sample")
plt.legend()
plt.show()

# Data preparation for plotting Residual Autocorrelation

y_true = y[(len(df1.index)-n_test_obs):(len(df1.index)),]
y_true = np.reshape(y_true,n_test_obs)
predictions_lm = np.reshape(predictions_lm, n_test_obs)
residuals = []
residuals = y_true - predictions_lm
plt.hist(residuals, bins = 50)
plt.ylabel('Residuals Histogram', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(6,2.5))
ax.scatter(residuals, predictions_lm)
#ax.scatter(residuals, y_true)
plt.legend()
plt.ylabel('Residual Autocorrelation', fontsize = 12)
plt.show()

print "\nPearson's statistic:\n", stats.pearsonr(y_true, predictions_lm)

###############################################################################

"""
 Information Ratio Calculation
"""    

info_ratio = []
info_ratio_is = []
info_ratio_wn = []

for m in [10,15,20,25]:
    excess_returns = []
    excess_returns_is = []
    excess_returns_wn = []
    excess_returns_std = []
    excess_returns_std_is = []
    excess_returns_std_wn = []
    idx = []
    idx_is = []
    for u in range(0,train_periods,1):
        idx.append(np.argsort(-prediction_set[u])[:m])
        pred_port = y[n_stocks*u+idx[u]]
        pred_port_ret = np.mean(pred_port)
        excess_returns.append((pred_port_ret - bench['Return'][u]))
        
        idx_is.append(np.argsort(-predictions_insample_set[u])[:m])
        pred_port_is = y[n_stocks*u+idx_is[u]]
        pred_port_ret_is = np.mean(pred_port_is)
        excess_returns_is.append((pred_port_ret_is - bench['Return'][u]))
        
        #White Noise IR
        wn = np.asarray(random.sample(range(1, n_stocks),m))
        pred_port_wn = y[n_stocks*u+wn]
        pred_port_ret_wn = np.mean(pred_port_wn)
        excess_returns_wn.append((pred_port_ret_wn - bench['Return'][u]))
        
    excess_returns_std.append(np.std(excess_returns))
    info_ratio.append(np.mean(excess_returns)/excess_returns_std)
    
    excess_returns_std_is.append(np.std(excess_returns_is))
    info_ratio_is.append(np.mean(excess_returns_is)/excess_returns_std_is)
    
    excess_returns_std_wn.append(np.std(excess_returns_wn))
    info_ratio_wn.append(np.mean(excess_returns_wn)/excess_returns_std_wn)
        
print "\nInformation Ratios:", info_ratio
print "\nInformation Ratios In-Sample:", info_ratio_is
print "\nInformation Ratios of White Noise:", info_ratio_wn

# Plotting Histograms of IR:

ir = pd.DataFrame(info_ratio, columns=['IR'])
ir['# of stocks'] = [10,15,20,25] 
plt.bar(ir['# of stocks'],ir['IR'], width = 4, tick_label=[10,15,20,25])
for a,b in zip(ir['# of stocks'], ir['IR']):
    plt.text(a, b, str(b)[0:6],horizontalalignment='center', fontsize=12, fontweight='bold' )
plt.xlabel('Number of Stocks', fontsize=12)
plt.ylabel('Information Ratio (0ut-Of-Sample)', fontsize=14)
axes = plt.gca()
axes.set_ylim([0,max(ir['IR'])+.1])
plt.show()

ir_is = pd.DataFrame(info_ratio_is, columns=['IR_is'])
ir_is['# of stocks'] = [10,15,20,25] 
plt.bar(ir_is['# of stocks'],ir_is['IR_is'], width = 4, tick_label=[10,15,20,25])
for a,b in zip(ir_is['# of stocks'], ir_is['IR_is']):
    plt.text(a, b, str(b)[0:6],horizontalalignment='center', fontsize=12, fontweight='bold' )
plt.xlabel('Number of Stocks', fontsize=12)
plt.ylabel('Information Ratio (In-Sample)', fontsize=14)
axes_is = plt.gca()
axes_is.set_ylim([0,max(ir_is['IR_is'])+.1])
plt.show()

ir_wn = pd.DataFrame(info_ratio_wn, columns=['IR_wn'])
ir_wn['# of stocks'] = [10,15,20,25] 
plt.bar(ir_wn['# of stocks'],ir_wn['IR_wn'], width = 4, tick_label=[10,15,20,25])
for a,b in zip(ir_wn['# of stocks'], ir_wn['IR_wn']):
    plt.text(a, b, str(b)[0:6],horizontalalignment='center', fontsize=12, fontweight='bold' )
plt.xlabel('Number of Stocks', fontsize=12)
plt.ylabel('Information Ratio (White Noise)', fontsize=14)
axes_wn = plt.gca()
axes_wn.set_ylim([0,0.30])
plt.show()


