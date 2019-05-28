# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Task 1 - Familiarization

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
plt.style.use('Solarize_Light2')
style.use('Solarize_Light2')
# %matplotlib inline
print(plt.style.available)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 60

df = pd.DataFrame(pd.read_csv('BATADAL_trainingset1.csv')) # No attacks
df_attacks = pd.DataFrame(pd.read_csv('BATADAL_trainingset2.csv')) # With attacks
df_nolabels = pd.DataFrame(pd.read_csv('BATADAL_test_dataset.csv')) # With attacks no labels
pd.set_option('display.expand_frame_repr', False)
df.describe()


# +
data_preproc = pd.DataFrame({
    'date': df["DATETIME"],
    'F_PU1': df["F_PU1"],
    'F_PU2': df["F_PU2"],
    'F_PU4': df["F_PU4"],
    'F_PU7': df["F_PU7"],
})[1000:1500]
data_preproc2 = pd.DataFrame({
    'date': df["DATETIME"],
    'L_T1': df["L_T1"],
    'L_T3': df["L_T3"],
    'L_T5': df["L_T5"],
})[1000:1500]
data_preproc3 = pd.DataFrame({
    'date': df["DATETIME"],
    'P_J280': df["P_J280"],
    'P_J256': df["P_J256"],
    'P_J302': df["P_J302"],
    'P_J14': df["P_J14"],
})[1000:1500]

data_preproc.plot(figsize=(20,10), x='date')
data_preproc2.plot(figsize=(20,10), x='date')
data_preproc3.plot(figsize=(20,10), x='date')
# -

# rcParams['figure.figsize'] = 12,10
# sns.heatmap(df.corr())
#
# values = df['F_PU1']
# rolling_mean = values.rolling(window=20).mean()
# rolling_mean2 = values.rolling(window=50).mean()
# plt.plot(df['DATETIME'], values, label='AMD')
# plt.plot(df['DATETIME'], rolling_mean, label='AMD 20 Day SMA', color='orange')
# plt.plot(df['DATETIME'], rolling_mean2, label='AMD 50 Day SMA', color='magenta')
# plt.legend(loc='upper left')
# plt.show()

# +
from numpy import mean
from sklearn.metrics import mean_squared_error

def moving_average_prediction(data, window = 3):
    test = [data[i] for i in range(window, len(data))]
    predictions = []
    
    current_prediction = window
    for t in range(len(test)):
        predicted_value = mean([data[i] for i in range(current_prediction-window,current_prediction)])
        predictions.append(predicted_value)
        current_prediction += 1
    # 	print('predicted=%f, expected=%f' % (yhat, obs))
    
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    return test, predictions

data, predictions = moving_average_prediction(df['F_PU1'].values, 3)

# plots
pd.DataFrame({"prediction":predictions[1000:2000],
            "actual": data[1000:2000]}).plot(figsize=(20,10))
# zoom plot
pd.DataFrame({"prediction":predictions[:100],
            "actual": data[:100]}).plot(figsize=(20,10))
# -

# # Task 2 - ARMA

# ARMA
# %pip install statsmodels scipy

import numpy as np
# from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

# ## Testing serial correlation with the Durbin-Watson statistic
# The DW statistic will lie in the 0-4 range, with a value near two indicating no first-order serial correlation. Positive serial correlation is associated with DW values below 2 and negative serial correlation with DW values above 2.
# The value of Durbin-Watson statistic is close to 2 if the errors are uncorrelated. 
#
# The Durbin-Watson statistic here is 0.0022. That means that there is a strong evidence that the variable has high positive autocorrelation.

# Durbin-Watson statistic. 
sm.stats.durbin_watson(df['F_PU1'])
sensors_to_model = ['F_PU1']

# ## Autocorrelation function
# We calculate the autocorrelation and partial autocorrelation functions to make an informed descision about what ARMA parameters to use.

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['F_PU1'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['F_PU1'], lags=40, ax=ax2)

# To determine the ARMA parameters we use the following rules of thumb:
# - Rule 1: If the ACF shows exponential decay, the PACF has a spike at lag 1, and no correlation for other lags, then use one autoregressive (p)parameter
# - Rule 2: If the ACF shows a sine-wave shape pattern or a set of exponential decays, the PACF has spikes at lags 1 and 2, and no correlation for other lags, the use two autoregressive (p) parameters
# - Rule 3: If the ACF has a spike at lag 1, no correlation for other lags, and the PACF damps out exponentially, then use one moving average (q) parameter.
# - Rule 4: If the ACF has spikes at lags 1 and 2, no correlation for other lags, and the PACF has a sine-wave shape pattern or a set of exponential decays, then use two moving average (q) parameter.
# - Rule 5: If the ACF shows exponential decay starting at lag 1, and the PACF shows exponential decay starting at lag 1, then use one autoregressive (p) and one moving average (q) parameter.
#
# Looking at the graphs above, we conclude that rule 2 seems to apply best to out data. Thus, we will use 2 autoregressive and no moving average parameters.

# +

def do_arma(train_series, test_series, params):
    print(f'####################################\nCurrent Series: {series.name}\n####################################')
    # Find optimal parameters based on AIC 
    best_params = params[0]
    lowest_aic = 999999999
    for param_set in params:
        arma_mod = sm.tsa.ARMA(train_series, param_set).fit()
        if arma_mod.aic < lowest_aic:
            lowest_aic = arma_mod.aic
            best_params = param_set
#         print(sm.stats.durbin_watson(arma_mod.resid.values))
#         print(arma_mod.params)
#         print(arma_mod.aic, arma_mod.bic, arma_mod.hqic)
        
    print('best params: ' + str(best_params))
    train_model = sm.tsa.ARMA(train_series, best_params).fit()
    test_model = ARIMA(test_series, best_params).fit(start_params = train_model.params, transpars = False, method='mle', trend='nc',disp=0)

    # The Durbin-Watson statistic is now very close to 2
    print(sm.stats.durbin_watson(arma.resid.values))

    #The equations are somewhat simpler if the time series is first reduced to zero-mean by subtracting the sample mean. Therefore, we will work with the mean-adjusted series
    # -

    # Plotting the residuals
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    resid = arma.resid
    ax = resid.plot(ax=ax);

    # +
    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    # -

    stats.normaltest(resid)

    # ## ARMA Model Autocorrelation
    print("Autocorrelation plots:")
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

    # ## Prediction

    prediction = arma.predict()
    pd.DataFrame({"prediction":prediction[100:600],
                "actual": train_series[100:600]}).plot(figsize=(20,10))
    return prediction
    # ## Anomaly detection
    # Use the parameters learned by the best model on Train set and apply it on test set
    std = np.std(test_model.resid)
    threshold = 2*std
    det_anom_lit = test_model.resid[test_model.resid > threshold]
    ind=[]
    tp=0
    fp=0
    for index, a in det_anom_lit.items():
        ind.append(index)
        if test_dataset.ATT_FLAG[index]==1:
            tp+=1
        else:
            fp+=1
    tn=test_dataset.loc[test_dataset.ATT_FLAG==-999].shape[0]-fp
    fn=test_dataset.loc[test_dataset.ATT_FLAG==1].shape[0]-tp
    Accuracy=100.0*(tp+tn)/(tp+tn+fp+fn)
    if (tp+fp)!=0:
        Precision=100.0*tp / (tp + fp)
    else:
        Precision=0
    Recall = 100.0*tp / (tp + fn)
    F_score = 100.0*2*tp /(2*tp + fp + fn)
    print ("TP:", tp)
    print ("FP:", fp)
    print("Accuracy: %.2f" % Accuracy)
    print("Precision: %.2f" % Precision)
    print("Recall: %.2f" %Recall)
    print("F_score: %.2f" % F_score)
    print('  ')


# +
def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


# +
param_sets = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
prediction = do_arma(df['P_J256'], df_attacks['P_J256'], param_sets)

print("MFE = ", mean_forecast_err(df['P_J256'], prediction))
print("MAE = ", mean_absolute_err(df['P_J256'], prediction))
print("MSE = ", mean_squared_error(df['P_J256'], prediction))


# +
do_arma(df['F_PU1'], (2,0))

print("MFE = ", mean_forecast_err(df['F_PU1'], prediction))
print("MAE = ", mean_absolute_err(df['F_PU1'], prediction))
print("MSE = ", mean_squared_error(df['F_PU1'], prediction))
# -

df_attacks['F_PU1']
# df_attacks['F_PU1']
