import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from sklearn.linear_model import LinearRegression
import math
from scipy.optimize import fsolve
import csv
import pandas as pd
from mpmath import *
import matplotlib.pyplot as plt
import argparse

def _compute_log_likelihood(params: tuple, *args: tuple) -> float:
    # Setting given parameters
    theta, mu, sigma = params
    X, dt = args
    n = len(X)
    # Calculating log likelihood
    sigma_tilde_squared = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    summation_term = sum((X[1:] - X[:-1] * np.exp(-theta * dt) - mu * (1 - np.exp(-theta * dt))) ** 2)
    summation_term = -summation_term / (2 * n * sigma_tilde_squared)
    log_likelihood = (-np.log(2 * np.pi) / 2) + (-np.log(np.sqrt(sigma_tilde_squared))) + summation_term
    return -log_likelihood

parser = argparse.ArgumentParser()
parser.add_argument("stock1")
parser.add_argument("stock2")
parser.add_argument("start")
parser.add_argument("end")
parser.add_argument("cost")
args = parser.parse_args()
stock1 = args.stock1
stock2 = args.stock2
data = yf.download(stock1 + " " + stock2 , start = args.start , end = args.end)["Close"].round(2)

regressor = LinearRegression()
a = np.array(np.log(data[stock1])).reshape(-1 , 1)
b = np.array(np.log(data[stock2])).reshape(-1 , 1)
regressor.fit(a , b)
beta = regressor.coef_[0][0]
print("beta = " , beta)
X_t = np.log(data[stock2]) - beta*np.log(data[stock1])

size = len(X_t)
df1 = pd.DataFrame(X_t) 
df1.to_csv('X_t.csv')

initial_guess = [np.std(X_t)/np.mean(X_t), np.mean(X_t), np.std(X_t)]
bounds = ((1e-5, None), (None, None), (1e-5, None))
test = minimize(_compute_log_likelihood, initial_guess, args=(X_t.values, 1), bounds=bounds)
theta = test.x[0]
mu = test.x[1]
sigma = test.x[2]
print("theta = " , theta , " mu = " ,  mu , " sigma = " , sigma)
Y_t = (X_t - mu) * math.sqrt(2 * theta) / sigma

df2 = pd.DataFrame(Y_t) 
df2.to_csv('Y_t.csv')
c = float(args.cost) * math.sqrt(2 * theta) / sigma

def summation(const , index):
    middle_term = lambda k: gamma((2 * k + 1) / 2) * ((math.sqrt(2) * const) ** (2 * k + index)) / fac(2 * k + index)
    term = nsum(middle_term , [0 , inf])
    return float(term)

def f(a):
    sum2 = summation(a[0] , 0)
    sum1 = summation(a[0] , 1)
    y = (sum1 / 2) - ((a[0] - c / 2) * sum2 / math.sqrt(2))
    return float(y)

a = fsolve(f , 0.7)
print("a = " , a)
b = -a
a_hat = a * (sigma / math.sqrt(2 * theta)) + mu
b_hat = b * (sigma / math.sqrt(2 * theta)) + mu

X = pd.read_csv('X_t.csv')
count = 0
numoftrade = 0
long = 0
short = 0
sell = 0
cover = 0
averageRate = []
rate1 = []
rate2 = []
bull = 0
stock1Return = 0
stock2Return = 0
cummax1 = []
cummax2 = []
cummaxAvg = []
dd1 = []
dd2 = []
ddAvg = []
mdd1 = 0
mdd2 = 0
mddAvg = 0

plt.plot(X.Date , X_t)
plt.axhline(y = a_hat , color='r', linestyle='-')
plt.axhline(y = b_hat , color='r', linestyle='-')

for i in range(size):
    if(X_t[i] < b_hat and count == 1):
        sell = data[stock1][i]
        cover = data[stock2][i]
        stock1Return = ((stock1Return + 100) * (sell / long) - 100).round(2)
        stock2Return = ((stock2Return + 100) * (2 - cover / short) - 100).round(2)
        long = 0
        short = 0
        count = 0
    elif(X_t[i] > a_hat and count == 2):
        cover = data[stock1][i]
        sell = data[stock2][i]
        stock1Return += ((1 - cover / short) * 100).round(2)
        stock2Return += ((sell / long - 1) * 100).round(2)
        long = 0
        short = 0
        count = 0
    if(X_t[i] > a_hat and count == 0):
        numoftrade += 1
        long = data[stock1][i]
        short = data[stock2][i]
        count = 1
        bull = 1
        rate1.append(((data[stock1][i] / long) * (stock1Return + 100) - 100).round(2))
        rate2.append(((2 - data[stock2][i] / short) * (stock2Return + 100) - 100).round(2))
        average = (rate2[len(rate2) - 1] + rate1[len(rate1) - 1] * beta) / (1 + beta)
        plt.scatter([X.Date[i]] , [X_t[i]] , s = 30 , c = 'lime' , alpha = 1)
    elif(X_t[i] < b_hat and count == 0):
        numoftrade += 1
        short = data[stock1][i]
        long = data[stock2][i]
        count = 2
        bull = 2
        rate1.append(((2 - data[stock1][i] / short) * (100 + stock1Return) - 100).round(2))
        rate2.append(((data[stock2][i] / long) * (100 + stock2Return) - 100).round(2))
        average = (rate2[len(rate2) - 1] + rate1[len(rate1) - 1] * beta) / (1 + beta)
        plt.scatter([X.Date[i]] , [X_t[i]] , s = 30 , c = 'lime' , alpha = 1)
    elif(count != 0):
        if(bull == 2):
            rate2.append(((data[stock2][i] / long) * (100 + stock2Return) - 100).round(2))
            rate1.append(((2 - data[stock1][i] / short) * (100 + stock1Return) - 100).round(2))
        else:
            rate1.append(((data[stock1][i] / long) * (100 + stock1Return) - 100).round(2))
            rate2.append(((2 - data[stock2][i] / short) * (100 + stock2Return) - 100).round(2))
        average = (rate2[len(rate2) - 1] + rate1[len(rate1) - 1] * beta) / (1 + beta)
    else:
        rate1.append(0)
        rate2.append(0)
        average = 0
    averageRate.append(average)
    if(i == 0):
        cummax1.append(rate1[0])
        cummax2.append(rate2[0])
        cummaxAvg.append(averageRate[0])
        dd1.append(0)
        dd2.append(0)
        ddAvg.append(0)
        continue
    if(cummax1[i - 1] < rate1[i]):
        cummax1.append(rate1[i])
    else:
        cummax1.append(cummax1[i - 1])
    if(cummax2[i - 1] < rate2[i]):
        cummax2.append(rate2[i])
    else:
        cummax2.append(cummax2[i - 1])
    if(cummaxAvg[i - 1] < averageRate[i]):
        cummaxAvg.append(averageRate[i])
    else:
        cummaxAvg.append(cummaxAvg[i - 1])
    dd1.append((rate1[i] + 100) / (cummax1[i] + 100) - 1)
    dd2.append((rate2[i] + 100) / (cummax2[i] + 100) - 1)
    ddAvg.append((averageRate[i] + 100) / (cummaxAvg[i] + 100) - 1)
    if(mdd1 > dd1[i]):
        mdd1 = dd1[i]
    if(mdd2 > dd2[i]):
        mdd2 = dd2[i]
    if(mddAvg > ddAvg[i]):
        mddAvg = ddAvg[i]

plt.show()

print("-------------------------------------------")
print("return for stock1 = " , rate1[len(rate1) - 1] , "%")
print("return for stock2 = " , rate2[len(rate2) - 1] , "%")
print("average return = " , averageRate[len(averageRate) - 1].round(2) , "%")
print("mdd1 = " , mdd1)
print("mdd2 = " , mdd2)
print("mddAvg = " , mddAvg)

plt.plot(X.Date , rate1 , label = stock1)
plt.plot(X.Date , rate2 , label = stock2)
plt.plot(X.Date , averageRate , label = "strategy")
plt.axhline(y = 0 , color = 'k', linestyle = '--' , linewidth = 1)
plt.legend()
plt.show()

plt.plot(X.Date , dd1 , label = stock1)
plt.plot(X.Date , dd2 , label = stock2)
plt.plot(X.Date , ddAvg , label = "strategy")
plt.legend()
plt.show()