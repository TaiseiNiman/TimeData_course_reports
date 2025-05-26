import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import yule_walker
from sklearn.metrics import mean_squared_error

# AR(3)係数（例）
true_a = np.array([0.9*(3**0.5), -0.81, 0.1])
n = 260
rim = 80
v = np.random.normal(0, 1, n)
y = np.zeros(n)
y[0:3] = v[0:3]  #  初期値

# データ生成（AR(3)プロセス）
for t in range(3, n):
    y[t] = true_a[0]*y[t-1] + true_a[1]*y[t-2] + true_a[2]*y[t-3] + v[t]

# Yule-Walker法でパラメータ推定
rho, sigma = yule_walker(y[0:n-rim], order=3)

# 予測値の計算
y_pred = np.zeros(n)
for t in range(3, n):
    if(t < n-rim): y_pred[t] = rho[0]*y[t-1] + rho[1]*y[t-2] + rho[2]*y[t-3]
    else: y_pred[t] = rho[0]*y_pred[t-1] + rho[1]*y_pred[t-2] + rho[2]*y_pred[t-3]
# RMSEの計算
rmse = np.sqrt(mean_squared_error(y[3:n-rim], y_pred[3:n-rim]))
print("推定係数:", rho)
print("推定分散:", sigma)
print("RMSE:", rmse)
rmse_forecast = np.sqrt(mean_squared_error(y[n-rim:], y_pred[n-rim:]))
print("Forecast RMSE:", rmse_forecast)

# プロット
plt.plot(y, label="Data", color='gray')
plt.plot(y_pred[:n-rim], label="Estimation", color='blue')
plt.plot(range(n-rim,n), y_pred[n-rim:], label="Prediction(last 100)", color='red')
plt.legend()
# plt.title("")
plt.xlabel("n")
plt.show()
