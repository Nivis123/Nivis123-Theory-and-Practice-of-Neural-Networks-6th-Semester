import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('household_power_consumption.txt', sep=';',
                 na_values=['?'], low_memory=False)

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('datetime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

df_hourly = df.resample('h').mean()
df_hourly = df_hourly.dropna()

df_hourly = df_hourly.loc['2007']

target = 'Global_active_power'
data_raw = df_hourly[[target]].values.flatten()

print(f"Всего часовых наблюдений: {len(data_raw)}")

class MinMaxScalerManual:
    def fit(self, data):
        self.min_ = data.min()
        self.max_ = data.max()

    def transform(self, data):
        return (data - self.min_) / (self.max_ - self.min_ + 1e-8)

    def inverse_transform(self, data_scaled):
        return data_scaled * (self.max_ - self.min_) + self.min_

scaler = MinMaxScalerManual()
scaler.fit(data_raw)
data = scaler.transform(data_raw)

def create_sequences(data, lookback=24, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback: i + lookback + horizon])
    return np.array(X), np.array(y)

lookback = 24
X_all, y_all = create_sequences(data, lookback)

print(f"Всего примеров: {X_all.shape[0]}")

train_size = int(0.7 * len(X_all))
val_size = int(0.15 * len(X_all))
test_size = len(X_all) - train_size - val_size

X_train, y_train = X_all[:train_size], y_all[:train_size]
X_val, y_val = X_all[train_size:train_size + val_size], y_all[train_size:train_size + val_size]
X_test, y_test = X_all[train_size + val_size:], y_all[train_size + val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def plot_results(y_true, y_pred, title, model_name, scaler):
    y_true_inv = scaler.inverse_transform(y_true.flatten())
    y_pred_inv = scaler.inverse_transform(y_pred.flatten())
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv, label='Факт', alpha=0.7)
    plt.plot(y_pred_inv, label='Прогноз', alpha=0.7)
    plt.title(f'{title} – {model_name}')
    plt.xlabel('Время (часы)')
    plt.ylabel('Активная мощность (кВт)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_losses(losses_dict, title):
    plt.figure(figsize=(10, 5))
    for name, loss in losses_dict.items():
        plt.plot(loss, label=name)
    plt.title(title)
    plt.xlabel('Эпоха')
    plt.ylabel('Потери (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

class RNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_ih = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

        self.W_ho = np.random.randn(output_size, hidden_size) * 0.01
        self.b_o = np.zeros((output_size, 1))

    def forward(self, x, h_prev):
        x = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)

        h = np.tanh(self.W_ih @ x + self.b_h + self.W_hh @ h_prev)
        y = self.W_ho @ h + self.b_o
        return h, y

    def backward(self, grad_h, grad_y, x, h_prev, h, cache):
        x = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)
        h = h.reshape(-1, 1)

        dW_ho = grad_y @ h.T
        db_o = grad_y

        grad_h_from_output = self.W_ho.T @ grad_y
        total_grad_h = grad_h + grad_h_from_output

        dtanh = (1 - h ** 2)
        grad_u = total_grad_h * dtanh

        dW_ih = grad_u @ x.T
        dW_hh = grad_u @ h_prev.T
        db_h = grad_u

        grad_h_prev = self.W_hh.T @ grad_u

        grads = {
            'W_ih': dW_ih,
            'W_hh': dW_hh,
            'b_h': db_h,
            'W_ho': dW_ho,
            'b_o': db_o
        }
        return grad_h_prev, grads

class GRUCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_z = np.random.randn(hidden_size, input_size) * 0.01
        self.U_z = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((hidden_size, 1))

        self.W_r = np.random.randn(hidden_size, input_size) * 0.01
        self.U_r = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_r = np.zeros((hidden_size, 1))

        self.W_h = np.random.randn(hidden_size, input_size) * 0.01
        self.U_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

        self.W_ho = np.random.randn(output_size, hidden_size) * 0.01
        self.b_o = np.zeros((output_size, 1))

    def forward(self, x, h_prev):
        x = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)

        z = 1 / (1 + np.exp(-(self.W_z @ x + self.b_z + self.U_z @ h_prev)))
        r = 1 / (1 + np.exp(-(self.W_r @ x + self.b_r + self.U_r @ h_prev)))

        h_tilde = np.tanh(self.W_h @ x + self.b_h + self.U_h @ (r * h_prev))

        h = (1 - z) * h_prev + z * h_tilde

        y = self.W_ho @ h + self.b_o

        cache = (x, h_prev, z, r, h_tilde, h)
        return h, y, cache

    def backward(self, grad_h, grad_y, cache):
        x, h_prev, z, r, h_tilde, h = cache

        dW_ho = grad_y @ h.T
        db_o = grad_y
        grad_h_from_output = self.W_ho.T @ grad_y
        total_grad_h = grad_h + grad_h_from_output

        grad_z = total_grad_h * (-h_prev + h_tilde)
        grad_h_prev_direct = total_grad_h * (1 - z)
        grad_h_tilde = total_grad_h * z

        d_tilde = (1 - h_tilde ** 2) * grad_h_tilde
        dW_h = d_tilde @ x.T
        db_h = d_tilde
        dU_h = d_tilde @ (r * h_prev).T
        grad_r_times_h_prev = self.U_h.T @ d_tilde

        grad_r_from_tilde = grad_r_times_h_prev * h_prev
        grad_h_prev_from_tilde = grad_r_times_h_prev * r

        lin_r = self.W_r @ x + self.b_r + self.U_r @ h_prev
        sig_r = r
        grad_lin_r = sig_r * (1 - sig_r) * grad_r_from_tilde
        dW_r = grad_lin_r @ x.T
        db_r = grad_lin_r
        dU_r = grad_lin_r @ h_prev.T
        grad_h_prev_from_r = self.U_r.T @ grad_lin_r

        lin_z = self.W_z @ x + self.b_z + self.U_z @ h_prev
        sig_z = z
        grad_lin_z = sig_z * (1 - sig_z) * grad_z
        dW_z = grad_lin_z @ x.T
        db_z = grad_lin_z
        dU_z = grad_lin_z @ h_prev.T
        grad_h_prev_from_z = self.U_z.T @ grad_lin_z

        grad_h_prev_total = (grad_h_prev_direct +
                             grad_h_prev_from_tilde +
                             grad_h_prev_from_r +
                             grad_h_prev_from_z)

        grads = {
            'W_z': dW_z, 'U_z': dU_z, 'b_z': db_z,
            'W_r': dW_r, 'U_r': dU_r, 'b_r': db_r,
            'W_h': dW_h, 'U_h': dU_h, 'b_h': db_h,
            'W_ho': dW_ho, 'b_o': db_o
        }
        return grad_h_prev_total, grads

class LSTMCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_i = np.random.randn(hidden_size, input_size) * 0.01
        self.U_i = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))

        self.W_f = np.random.randn(hidden_size, input_size) * 0.01
        self.U_f = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))

        self.W_o = np.random.randn(hidden_size, input_size) * 0.01
        self.U_o = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))

        self.W_c = np.random.randn(hidden_size, input_size) * 0.01
        self.U_c = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))

        self.W_ho = np.random.randn(output_size, hidden_size) * 0.01
        self.b_out = np.zeros((output_size, 1))

    def forward(self, x, h_prev, c_prev):
        x = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)
        c_prev = c_prev.reshape(-1, 1)

        i = 1 / (1 + np.exp(-(self.W_i @ x + self.b_i + self.U_i @ h_prev)))
        f = 1 / (1 + np.exp(-(self.W_f @ x + self.b_f + self.U_f @ h_prev)))
        o = 1 / (1 + np.exp(-(self.W_o @ x + self.b_o + self.U_o @ h_prev)))
        c_tilde = np.tanh(self.W_c @ x + self.b_c + self.U_c @ h_prev)

        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)

        y = self.W_ho @ h + self.b_out

        cache = (x, h_prev, c_prev, i, f, o, c_tilde, c, h)
        return h, c, y, cache

    def backward(self, grad_h, grad_c, grad_y, cache):
        x, h_prev, c_prev, i, f, o, c_tilde, c, h = cache

        dW_ho = grad_y @ h.T
        db_out = grad_y
        grad_h_from_output = self.W_ho.T @ grad_y
        total_grad_h = grad_h + grad_h_from_output

        grad_o_from_h = total_grad_h * np.tanh(c)
        grad_c_from_h = total_grad_h * o * (1 - np.tanh(c) ** 2)

        total_grad_c = grad_c + grad_c_from_h

        lin_o = self.W_o @ x + self.b_o + self.U_o @ h_prev
        sig_o = o
        grad_lin_o = sig_o * (1 - sig_o) * grad_o_from_h
        dW_o = grad_lin_o @ x.T
        db_o_vec = grad_lin_o
        dU_o = grad_lin_o @ h_prev.T
        grad_h_prev_from_o = self.U_o.T @ grad_lin_o

        grad_f = total_grad_c * c_prev
        grad_i = total_grad_c * c_tilde
        grad_c_tilde = total_grad_c * i
        grad_c_prev = total_grad_c * f

        lin_f = self.W_f @ x + self.b_f + self.U_f @ h_prev
        sig_f = f
        grad_lin_f = sig_f * (1 - sig_f) * grad_f
        dW_f = grad_lin_f @ x.T
        db_f = grad_lin_f
        dU_f = grad_lin_f @ h_prev.T
        grad_h_prev_from_f = self.U_f.T @ grad_lin_f

        lin_i = self.W_i @ x + self.b_i + self.U_i @ h_prev
        sig_i = i
        grad_lin_i = sig_i * (1 - sig_i) * grad_i
        dW_i = grad_lin_i @ x.T
        db_i = grad_lin_i
        dU_i = grad_lin_i @ h_prev.T
        grad_h_prev_from_i = self.U_i.T @ grad_lin_i

        lin_c = self.W_c @ x + self.b_c + self.U_c @ h_prev
        dtanh_c = (1 - c_tilde ** 2) * grad_c_tilde
        dW_c = dtanh_c @ x.T
        db_c = dtanh_c
        dU_c = dtanh_c @ h_prev.T
        grad_h_prev_from_c = self.U_c.T @ dtanh_c

        grad_h_prev = (grad_h_prev_from_o + grad_h_prev_from_f +
                       grad_h_prev_from_i + grad_h_prev_from_c)

        grads = {
            'W_i': dW_i, 'U_i': dU_i, 'b_i': db_i,
            'W_f': dW_f, 'U_f': dU_f, 'b_f': db_f,
            'W_o': dW_o, 'U_o': dU_o, 'b_o': db_o_vec,
            'W_c': dW_c, 'U_c': dU_c, 'b_c': db_c,
            'W_ho': dW_ho, 'b_out': db_out
        }
        return grad_h_prev, grad_c_prev, grads

def train_manual_model(cell, X_train, y_train, X_val, y_val,
                       epochs=20, lr=0.001, print_every=5):
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_params = None

    is_lstm = isinstance(cell, LSTMCell)

    for epoch in range(epochs):
        epoch_train_loss = 0
        for idx in range(len(X_train)):
            x_seq = X_train[idx]
            target = y_train[idx]

            h = np.zeros((cell.hidden_size, 1))
            if is_lstm:
                c = np.zeros((cell.hidden_size, 1))

            h_states = [h]
            if is_lstm:
                c_states = [c]
            cache_list = []

            for t in range(len(x_seq)):
                x_t = x_seq[t].reshape(-1, 1)
                if is_lstm:
                    h, c, y_pred, cache = cell.forward(x_t, h_states[-1], c_states[-1])
                    h_states.append(h)
                    c_states.append(c)
                    cache_list.append(cache)
                elif cell.__class__.__name__ == 'GRUCell':
                    h, y_pred, cache = cell.forward(x_t, h_states[-1])
                    h_states.append(h)
                    cache_list.append(cache)
                else:
                    h, y_pred = cell.forward(x_t, h_states[-1])
                    h_states.append(h)
                    cache_list.append((x_t, h_states[-2], h))

            y_pred_last = y_pred
            loss = (y_pred_last - target) ** 2
            epoch_train_loss += loss.item()

            grad_y_final = 2 * (y_pred_last - target)

            grads_acc = {k: np.zeros_like(v) for k, v in cell.__dict__.items() if isinstance(v, np.ndarray)}

            if is_lstm:
                grad_h_next = np.zeros((cell.hidden_size, 1))
                grad_c_next = np.zeros((cell.hidden_size, 1))
                for t in reversed(range(len(x_seq))):
                    current_grad_y = grad_y_final if t == len(x_seq)-1 else np.zeros_like(grad_y_final)
                    cache = cache_list[t]
                    grad_h_next, grad_c_next, grads_t = cell.backward(grad_h_next, grad_c_next, current_grad_y, cache)
                    for k, g in grads_t.items():
                        grads_acc[k] += g
            else:
                grad_h_next = np.zeros((cell.hidden_size, 1))
                for t in reversed(range(len(x_seq))):
                    current_grad_y = grad_y_final if t == len(x_seq)-1 else np.zeros_like(grad_y_final)
                    if cell.__class__.__name__ == 'GRUCell':
                        cache = cache_list[t]
                        grad_h_next, grads_t = cell.backward(grad_h_next, current_grad_y, cache)
                    else:
                        x_t, h_prev, h_cur = cache_list[t]
                        grad_h_next, grads_t = cell.backward(grad_h_next, current_grad_y, x_t, h_prev, h_cur, None)
                    for k, g in grads_t.items():
                        grads_acc[k] += g

            clip_thresh = 5.0
            for k in grads_acc:
                np.clip(grads_acc[k], -clip_thresh, clip_thresh, out=grads_acc[k])

            for name, param in cell.__dict__.items():
                if name in grads_acc:
                    param -= lr * grads_acc[name]

        avg_train_loss = epoch_train_loss / len(X_train)
        train_losses.append(avg_train_loss)

        val_preds = []
        for idx in range(len(X_val)):
            x_seq = X_val[idx]
            h = np.zeros((cell.hidden_size, 1))
            if is_lstm:
                c = np.zeros((cell.hidden_size, 1))
            for t in range(len(x_seq)):
                x_t = x_seq[t].reshape(-1, 1)
                if is_lstm:
                    h, c, y_pred, _ = cell.forward(x_t, h, c)
                elif cell.__class__.__name__ == 'GRUCell':
                    h, y_pred, _ = cell.forward(x_t, h)
                else:
                    h, y_pred = cell.forward(x_t, h)
            val_preds.append(y_pred.flatten()[0])
        val_preds = np.array(val_preds).reshape(-1, 1)
        val_loss = np.mean((val_preds - y_val) ** 2)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {k: v.copy() for k, v in cell.__dict__.items() if isinstance(v, np.ndarray)}

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if best_params:
        for k, v in best_params.items():
            cell.__dict__[k] = v

    return train_losses, val_losses






input_size = 1
hidden_size = 32
output_size = 1
epochs_manual = 10
lr = 0.01
lr_lib = 0.001

print("\n" + "=" * 50)
print("Обучение ручной RNN")
rnn_cell = RNNCell(input_size, hidden_size, output_size)
rnn_train_loss, rnn_val_loss = train_manual_model(rnn_cell, X_train, y_train, X_val, y_val,
                                                  epochs=epochs_manual, lr=lr, print_every=2)

print("\n" + "=" * 50)
print("Обучение ручной GRU")
gru_cell = GRUCell(input_size, hidden_size, output_size)
gru_train_loss, gru_val_loss = train_manual_model(gru_cell, X_train, y_train, X_val, y_val,
                                                  epochs=epochs_manual, lr=lr, print_every=2)

print("\n" + "=" * 50)
print("Обучение ручной LSTM")
lstm_cell = LSTMCell(input_size, hidden_size, output_size)
lstm_train_loss, lstm_val_loss = train_manual_model(lstm_cell, X_train, y_train, X_val, y_val,
                                                    epochs=epochs_manual, lr=lr, print_every=2)

manual_losses = {
    'RNN_train': rnn_train_loss,
    'RNN_val': rnn_val_loss,
    'GRU_train': gru_train_loss,
    'GRU_val': gru_val_loss,
    'LSTM_train': lstm_train_loss,
    'LSTM_val': lstm_val_loss
}

X_train_t = torch.tensor(X_train).unsqueeze(-1)
y_train_t = torch.tensor(y_train)
X_val_t = torch.tensor(X_val).unsqueeze(-1)
y_val_t = torch.tensor(y_val)
X_test_t = torch.tensor(X_test).unsqueeze(-1)
y_test_t = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_library_model(model, train_loader, val_loader, epochs=20, lr=0.001, print_every=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return train_losses, val_losses

print("\n" + "=" * 50)
print("Обучение библиотечной RNN")
rnn_lib = RNNModel(input_size, hidden_size, output_size)
rnn_lib_train, rnn_lib_val = train_library_model(rnn_lib, train_loader, val_loader, epochs=epochs_manual, lr=lr_lib,
                                                 print_every=2)

print("\n" + "=" * 50)
print("Обучение библиотечной GRU")
gru_lib = GRUModel(input_size, hidden_size, output_size)
gru_lib_train, gru_lib_val = train_library_model(gru_lib, train_loader, val_loader, epochs=epochs_manual, lr=lr_lib,
                                                 print_every=2)

print("\n" + "=" * 50)
print("Обучение библиотечной LSTM")
lstm_lib = LSTMModel(input_size, hidden_size, output_size)
lstm_lib_train, lstm_lib_val = train_library_model(lstm_lib, train_loader, val_loader, epochs=epochs_manual, lr=lr_lib,
                                                   print_every=2)

def evaluate_manual(cell, X_test, y_test, is_lstm=False):
    preds = []
    for idx in range(len(X_test)):
        x_seq = X_test[idx]
        h = np.zeros((cell.hidden_size, 1))
        if is_lstm:
            c = np.zeros((cell.hidden_size, 1))
        for t in range(len(x_seq)):
            x_t = x_seq[t].reshape(-1, 1)
            if is_lstm:
                h, c, y_pred, _ = cell.forward(x_t, h, c)
            elif cell.__class__.__name__ == 'GRUCell':
                h, y_pred, _ = cell.forward(x_t, h)
            else:
                h, y_pred = cell.forward(x_t, h)
        preds.append(y_pred.flatten()[0])
    return np.array(preds).reshape(-1, 1)

def evaluate_library(model, loader):
    device = next(model.parameters()).device
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())
            targets.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(targets)

y_pred_rnn_man = evaluate_manual(rnn_cell, X_test, y_test)
y_pred_gru_man = evaluate_manual(gru_cell, X_test, y_test, is_lstm=False)
y_pred_lstm_man = evaluate_manual(lstm_cell, X_test, y_test, is_lstm=True)

y_pred_rnn_lib, y_test_lib = evaluate_library(rnn_lib, test_loader)
y_pred_gru_lib, _ = evaluate_library(gru_lib, test_loader)
y_pred_lstm_lib, _ = evaluate_library(lstm_lib, test_loader)

models = {
    'RNN (ручная)': y_pred_rnn_man,
    'GRU (ручная)': y_pred_gru_man,
    'LSTM (ручная)': y_pred_lstm_man,
    'RNN (библ.)': y_pred_rnn_lib,
    'GRU (библ.)': y_pred_gru_lib,
    'LSTM (библ.)': y_pred_lstm_lib
}

print("\n" + "=" * 50)
print("МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 50)
for name, pred in models.items():
    if pred.shape != y_test.shape:
        pred = pred.reshape(y_test.shape)
    mae_val = mae(y_test, pred)
    rmse_val = rmse(y_test, pred)
    mape_val = mape(y_test, pred)
    print(f"{name}: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, MAPE={mape_val:.2f}%")

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(manual_losses['RNN_train'], label='RNN train')
plt.plot(manual_losses['RNN_val'], label='RNN val')
plt.title('Ручная RNN')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(manual_losses['GRU_train'], label='GRU train')
plt.plot(manual_losses['GRU_val'], label='GRU val')
plt.title('Ручная GRU')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(manual_losses['LSTM_train'], label='LSTM train')
plt.plot(manual_losses['LSTM_val'], label='LSTM val')
plt.title('Ручная LSTM')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(rnn_lib_train, label='RNN train')
plt.plot(rnn_lib_val, label='RNN val')
plt.title('Библиотечная RNN')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(gru_lib_train, label='GRU train')
plt.plot(gru_lib_val, label='GRU val')
plt.title('Библиотечная GRU')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(lstm_lib_train, label='LSTM train')
plt.plot(lstm_lib_val, label='LSTM val')
plt.title('Библиотечная LSTM')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

n_show = 200
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(scaler.inverse_transform(y_test[:n_show]), label='Факт')
plt.plot(scaler.inverse_transform(y_pred_rnn_man[:n_show]), label='Ручная RNN')
plt.title('Ручная RNN')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(scaler.inverse_transform(y_test[:n_show]), label='Факт')
plt.plot(scaler.inverse_transform(y_pred_rnn_lib[:n_show]), label='Библ. RNN')
plt.title('Библиотечная RNN')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(scaler.inverse_transform(y_test[:n_show]), label='Факт')
plt.plot(scaler.inverse_transform(y_pred_gru_man[:n_show]), label='Ручная GRU')
plt.title('Ручная GRU')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(scaler.inverse_transform(y_test[:n_show]), label='Факт')
plt.plot(scaler.inverse_transform(y_pred_gru_lib[:n_show]), label='Библ. GRU')
plt.title('Библиотечная GRU')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(scaler.inverse_transform(y_test[:n_show]), label='Факт')
plt.plot(scaler.inverse_transform(y_pred_lstm_man[:n_show]), label='Ручная LSTM')
plt.title('Ручная LSTM')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(scaler.inverse_transform(y_test[:n_show]), label='Факт')
plt.plot(scaler.inverse_transform(y_pred_lstm_lib[:n_show]), label='Библ. LSTM')
plt.title('Библиотечная LSTM')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

