import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
# ① WAVファイル読み込み（ファイルパスを適宜変更）
sample_rate, data = wavfile.read("Canon-and-Gigue-in-D-major.wav")

# ② ステレオ→モノラル変換（2チャンネルの場合は平均）
if len(data.shape) == 2:
    data = data.mean(axis=1)

# ③ 正規化：振幅を最大値で割る（-1.0〜1.0にスケーリング）
normalized_data = data / np.max(np.abs(data))

# ④ 可視化：一部の時系列をプロット（長いので最初の数秒）
# print(sample_rate)
# N = 1  # 
# plt.plot(np.arange(len(data)) / sample_rate, normalized_data[0::int(N)]**2) #音のエネルギー
# plt.title("Normalized Squared Amplitude of Canon-and-Gigue-in-D-major y[n]")
# plt.xlabel(f"Time (seconds)")
# plt.ylabel("y")
# plt.grid(True)
# plt.show()
# plt.scatter(normalized_data[sample_rate:]**2, normalized_data[:-sample_rate]**2, alpha=0.5, s=1)
# plt.xlabel(f"y[n - {sample_rate}]")   # 横軸のラベルを変更
# plt.ylabel("y[n]")          # 縦軸のラベルを変更
# plt.title(f"Scatter Plot of (y[n], y[n - {sample_rate}])") #y[n−sample_rate] is the squared amplitude of the sound one second earlier.
# plt.grid(True)
# plt.show()
# plt.hist(normalized_data**2, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black',log=False)
# plt.xlabel("Normalized Squared Amplitude")   # 横軸のラベルを変更
# plt.ylabel("Density")          # 縦軸のラベルを変更
# plt.title("Histogram of Normalized Squared Amplitude 50 bins") #y[n−sample_rate] is the squared amplitude of the sound one second earlier.
# plt.grid(True)
# plt.show()

# sample autocorrelation
def autocorrelation(x, lag):
    n = len(x)
    mean = np.mean(x)
    c = np.correlate(x - mean, x - mean, mode='full')
    return c[n - 1 + lag] / (n - lag)  # 正規化を行う

# sample autocorrelation　自作
def sample_autocorrelation(x, lag):#xは一次元配列 sharp(1,), lagは整数
    n = len(x)
    mean = np.mean(x)
    var = np.mean((x[:]-mean)**2)
    # r = np.sum((x[:n-lag]-mean)*(x[lag:n]-mean))/n/var
    c = np.mean((x[:n-lag]-mean)*(x[lag:n]-mean))
    return c


# sample autocorrelationを各lagで計算しプロット
# autocorrelation = []
# for lag in range(0,len(normalized_data),sample_rate):
#     autocorrelation.append(sample_autocorrelation(normalized_data**2, lag))
# plt.plot(np.arange(0, len(normalized_data), sample_rate)/sample_rate, autocorrelation)    
# plt.title("Sample Autocorrelation of Canon-and-Gigue-Sin-D-major y[n]")
# plt.xlabel("Lag k per sample rate")
# plt.ylabel("R_k")
# plt.grid(True)
# plt.show()

# sample periodogramのplot
# ck = []
# for lag in range(0,len(normalized_data),sample_rate):
#         ck.append(sample_autocorrelation(normalized_data**2, lag))

# k = np.arange(sample_rate, len(normalized_data), sample_rate)/sample_rate
# fj = np.arange(1,int(len(k)/2))/len(normalized_data)
# cos = np.cos(2*np.pi*k[:,]*fj)
# pj= ck[0] + 2*np.dot(ck[1:], cos) #cos[0,:]は定数項なので除外
# plt.plot(fj, pj)
# plt.title("Sample Periodogram of Canon-and-Gigue-Sin-D-major y[n]")    
# plt.xlabel("f (Hz) per sample rate")
# plt.ylabel("P(f)")
# plt.grid(True)
# plt.show()  

# ck = []
# for lag in range(0, len(normalized_data), sample_rate):
#     ck.append(sample_autocorrelation(normalized_data**2, lag))
# ck = np.array(ck)

# # ラグ秒数：k=1,2,... 秒
# k = np.arange(1, len(ck))  # 長さ = len(ck) - 1

# # 周波数軸の定義（Hz）
# N = len(normalized_data)
# fs = sample_rate
# fj = np.arange(0, int(len(ck)/2)) * fs / N 

# # cos 行列： shape = (len(k), len(fj))
# cos = np.cos(2 * np.pi * k[:, np.newaxis] * fj[np.newaxis, :])

# # ピリオドグラム計算
# pj = ck[0] + 2 * np.dot(ck[1:], cos) # ck[0]は定数項なので除外

# # プロット
# plt.plot(fj, pj)
# plt.title("Sample Periodogram of Canon-and-Gigue-in-D-major y[n]")    
# plt.xlabel("Frequency")
# plt.ylabel("P(f)")
# plt.grid(True)
# plt.show()

# 指数分布の課題
# N = 100  # サンプル数
# lamda = 2.0  # 平均値の逆数
samples = np.random.exponential(scale=0.5, size=(10,100))  # 指数分布に従う乱数を生成
λ = np.linspace(0.01, 10, 1000)
def l(λ):
    return 100*np.log(λ) - λ*100*np.mean(samples[0,:])  # 対数尤度関数の定義
y = l(λ)  # 対数尤度関数の計算
plt.plot(λ, y)
plt.title("Log-Likelihood Function of Exponential Distribution")
plt.xlabel("λ")
plt.ylabel("l(λ)")
plt.grid(True)
plt.show()
# 最尤推定によって得られたλの値の分布をヒストグラムで表示
# 最尤推定値λ̂のヒストグラムを表示
lambda_hats = 1/np.mean(samples, axis=1)
plt.hist(lambda_hats, bins=15, density=False, alpha=0.7, color='blue', edgecolor='black')
plt.title("Histogram of Maximum Likelihood Estimates (λ̂)")
plt.xlabel("λ̂")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()