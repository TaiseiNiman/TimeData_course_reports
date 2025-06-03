##### Exercise(6) example (改良版)
##### (c) G.Tanaka @ NITech + 改変

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

np.random.seed(0)
torch.manual_seed(0)

# データの読み込み（sunspot.year）
data_sunspot = sm.datasets.get_rdataset('sunspot.year')
sunspot = np.array(data_sunspot.data.value).reshape(-1,1)
month = np.array(data_sunspot.data.time).reshape(-1,1)

# 訓練・テストデータに分割
train_ratio = 0.9
len_train = int(train_ratio*len(sunspot))
s_train = sunspot[:len_train]
s_test = sunspot[len_train:]

# 時系列データを教師データに変換（lookback: 過去の窓幅）
def transform_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i+lookback])
        y.append(dataset[i+1:i+lookback+1])
    return torch.tensor(X).float(), torch.tensor(y).float()

lookback = 3
X_train, y_train = transform_dataset(s_train, lookback)
X_test, y_test = transform_dataset(s_test, lookback)

# RNNモデル（ReLU）
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=50, num_layers=1,
                          batch_first=True, nonlinearity='relu')
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

# LSTMモデル
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1,
                            batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

# 学習と評価
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name=""):
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    for epoch in range(1000):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_rmse = torch.sqrt(loss_fn(model(X_train), y_train)).item()
                test_rmse = torch.sqrt(loss_fn(model(X_test), y_test)).item()
                print(f"[{model_name}] Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

    # 最終RMSE返す
    model.eval()
    with torch.no_grad():
        train_rmse = torch.sqrt(loss_fn(model(X_train), y_train)).item()
        test_rmse = torch.sqrt(loss_fn(model(X_test), y_test)).item()
    return model, train_rmse, test_rmse

# プロット関数
def plot_prediction(model, X_train, X_test, model_name):
    with torch.no_grad():
        train_plot = np.ones(len(sunspot)) * np.nan  # → shape: (N,)
        test_plot = np.ones(len(sunspot)) * np.nan
        train_pred = model(X_train)[:, -1, :].squeeze().numpy()
        test_pred = model(X_test)[:, -1, :].squeeze().numpy()
        train_plot[lookback:len_train] = train_pred
        test_plot[len_train+lookback:] = test_pred

    plt.figure(figsize=(10,5))
    plt.plot(sunspot.squeeze(), color='gray', label='Observed Data')
    plt.plot(train_plot, label=f'{model_name} Prediction (Train)', color='C0')
    plt.plot(test_plot, label=f'{model_name} Prediction (Test)', color='C3')
    plt.xlabel("n")
    plt.ylabel("Sunspot Count")
    plt.title(f"Sunspot Prediction using {model_name}")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# ------------------------------
# RNNモデル（ReLU）による実行
rnn_model = RNNModel()
rnn_model, rnn_train_rmse, rnn_test_rmse = train_and_evaluate(
    rnn_model, X_train, y_train, X_test, y_test, model_name="RNN (ReLU)")
plot_prediction(rnn_model, X_train, X_test, model_name="RNN (ReLU)")

# LSTMモデルによる実行
lstm_model = LSTMModel()
lstm_model, lstm_train_rmse, lstm_test_rmse = train_and_evaluate(
    lstm_model, X_train, y_train, X_test, y_test, model_name="LSTM")
plot_prediction(lstm_model, X_train, X_test, model_name="LSTM")

# ------------------------------
# 最終的なRMSE結果
print("\n======== Final RMSE Results ========")
print(f"RNN (ReLU) Test RMSE : {rnn_test_rmse:.4f}")
print(f"LSTM       Test RMSE : {lstm_test_rmse:.4f}")
