##### Exercise(5) revised for RMSE minimization
##### (c) Modified from G.Tanaka @ NITech

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(0)

# データの読み込み（sunspot.year）
data_sunspot = sm.datasets.get_rdataset('sunspot.year')
y = data_sunspot.data.value
month = data_sunspot.data.time

# トレーニングデータとテストデータに分割
train_ratio = 0.9
len_train = int(train_ratio * len(y))
len_test = len(y) - len_train
y_train = y[:len_train]
y_test = y[len_train:]

# RMSE最小化に基づく最適mの探索
rmse_min = 1e+10
m_max = 15
for m in range(1, m_max + 1):
    model = sm.tsa.statespace.SARIMAX(y_train, trend='n', order=(m, 0, 1))
    results = model.fit(disp=False)

    # テスト期間の予測
    y_pred_test = results.get_forecast(len_test)
    y_pred_test_mean = y_pred_test.predicted_mean
    SE = [(y_test.iloc[i] - y_pred_test_mean.iloc[i])**2 for i in range(len_test)]
    RMSE = np.sqrt(np.mean(SE))

    print(f'RMSE for ARMA({m},1) = {RMSE:.4f}')

    if RMSE < rmse_min:
        m_opt = m
        rmse_min = RMSE
        best_result = results

print(f'[m_opt, RMSE_min] = [{m_opt}, {rmse_min:.4f}]')

# 最適モデルによる再予測（トレーニング＋テスト）
# トレーニング期間の予測
y_pred_train = best_result.get_prediction()
y_pred_train_mean = y_pred_train.predicted_mean
y_pred_train_ci = y_pred_train.conf_int()

# テスト期間の予測
y_pred_test = best_result.get_forecast(len_test)
y_pred_test_mean = y_pred_test.predicted_mean
y_pred_test_ci = y_pred_test.conf_int()

# 描画
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(y)), y, color='gray', label='Observed data')
plt.plot(np.arange(len_train), y_pred_train_mean, color='C0', label='Prediction (train)')
plt.fill_between(np.arange(len_train), y_pred_train_ci.iloc[:,0], y_pred_train_ci.iloc[:,1], color='C0', alpha=.2, label='Confidence interval (train)')
plt.plot(np.arange(len_train, len(y)), y_pred_test_mean, color='C3', label='Prediction (test)')
plt.fill_between(np.arange(len_train, len(y)), y_pred_test_ci.iloc[:,0], y_pred_test_ci.iloc[:,1], color='C3', alpha=.2, label='Confidence interval (test)')
plt.xlabel('Year (n)')
plt.ylabel('Sunspot count')
plt.title(f'Sunspot Prediction by ARMA({m_opt},1)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
