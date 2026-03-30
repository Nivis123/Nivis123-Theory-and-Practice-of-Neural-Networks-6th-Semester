import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, \
    classification_report
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings

warnings.filterwarnings('ignore')


# Собственная реализация многослойного персептрона (MLP)


class MLP:
    def __init__(self, layers, activations, learning_rate=0.01, epochs=100, batch_size=32, random_state=42):

        self.layers = layers
        self.activations = activations
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        np.random.seed(random_state)

        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            if activations[i] == 'relu':
                std = np.sqrt(2. / layers[i])
            else:
                std = 0.1
            w = np.random.randn(layers[i], layers[i + 1]) * std
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _activate(self, z, name, derivative=False):
        if name == 'relu':
            if derivative:
                return (z > 0).astype(float)
            return np.maximum(0, z)
        elif name == 'linear':
            if derivative:
                return np.ones_like(z)
            return z
        elif name == 'softmax':
            if derivative:
                raise NotImplementedError("Производная softmax не вызывается напрямую")
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Неизвестная активация: {name}")

    def _forward(self, X):
        A = X
        A_list = [A]
        Z_list = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            Z = np.dot(A, w) + b
            Z_list.append(Z)
            A = self._activate(Z, self.activations[i])
            A_list.append(A)
        return A_list, Z_list

    def _compute_loss(self, y_true, y_pred, loss_type):
        if loss_type == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif loss_type == 'crossentropy':
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            raise ValueError("Неизвестный тип потерь")

    def fit(self, X, y, loss_type='mse', X_val=None, y_val=None, verbose=True):

        n_samples = X.shape[0]
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            #!
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]


                A_list, Z_list = self._forward(X_batch)
                y_pred = A_list[-1]


                m_batch = X_batch.shape[0]

                if loss_type == 'mse':
                    dZ = 2.0 * (y_pred - y_batch) / m_batch
                elif loss_type == 'crossentropy':
                    dZ = (y_pred - y_batch) / m_batch
                else:
                    raise ValueError("Неизвестный тип потерь")

                dW = [None] * len(self.weights)
                dB = [None] * len(self.biases)

                dW[-1] = np.dot(A_list[-2].T, dZ)
                dB[-1] = np.sum(dZ, axis=0, keepdims=True)

                for i in range(len(self.weights) - 2, -1, -1):
                    dA_next = np.dot(dZ, self.weights[i + 1].T)
                    dZ = dA_next * self._activate(Z_list[i], self.activations[i], derivative=True)
                    dW[i] = np.dot(A_list[i].T, dZ)
                    dB[i] = np.sum(dZ, axis=0, keepdims=True)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * dW[i]
                    self.biases[i] -= self.lr * dB[i]

            A_train, _ = self._forward(X)
            y_train_pred = A_train[-1]
            train_loss = self._compute_loss(y, y_train_pred, loss_type)
            history['train_loss'].append(train_loss)

            if X_val is not None:
                A_val, _ = self._forward(X_val)
                y_val_pred = A_val[-1]
                val_loss = self._compute_loss(y_val, y_val_pred, loss_type)
                history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Train loss: {train_loss:.6f}" +
                      (f", Val loss: {val_loss:.6f}" if X_val is not None else ""))

        return history

    def predict(self, X):
        A, _ = self._forward(X)
        return A[-1]


# 2.1 МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН-РЕГРЕССОР (Laptop_price)

print("=" * 70)
print("2.1 РЕГРЕССИЯ: ПРЕДСКАЗАНИЕ ЦЕНЫ НОУТБУКА")
print("=" * 70)


df_laptop = pd.read_csv('Laptop_price.csv')
print("\nПервые 5 строк:")
print(df_laptop.head())

# Кодирууем
brand_encoded = pd.get_dummies(df_laptop['Brand'], prefix='Brand')
X = pd.concat([brand_encoded, df_laptop[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]],
              axis=1)
y = df_laptop['Price'].values.reshape(-1, 1)

#Маштабируем признаки
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Сначала 80% train, 20% test потом из train выделяем 20% на валидацию ( 64% train, 16% val, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\nРазмеры: X_train {X_train.shape}, y_train {y_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")


# Собственная реализация MLP-регрессора

print("\n--- Обучение собственного MLP-регрессора ---")
mlp_reg_own = MLP(layers=[X_train.shape[1], 64, 32, 1],
                  activations=['relu', 'relu', 'linear'],
                  learning_rate=0.001, epochs=200, batch_size=32, random_state=42)
history_own = mlp_reg_own.fit(X_train, y_train, loss_type='mse', X_val=X_val, y_val=y_val, verbose=True)


y_train_pred_own = scaler_y.inverse_transform(mlp_reg_own.predict(X_train))
y_val_pred_own = scaler_y.inverse_transform(mlp_reg_own.predict(X_val))
y_test_pred_own = scaler_y.inverse_transform(mlp_reg_own.predict(X_test))
y_train_true = scaler_y.inverse_transform(y_train)
y_val_true = scaler_y.inverse_transform(y_val)
y_test_true = scaler_y.inverse_transform(y_test)



def calc_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, mae, r2


mse_train, mae_train, r2_train = calc_metrics(y_train_true, y_train_pred_own)
mse_val, mae_val, r2_val = calc_metrics(y_val_true, y_val_pred_own)
mse_test, mae_test, r2_test = calc_metrics(y_test_true, y_test_pred_own)

print("\nРезультаты собственного MLP (регрессия):")
print(f"  Train: MSE = {mse_train:.2f}, MAE = {mae_train:.2f}, R2 = {r2_train:.4f}")
print(f"  Val:   MSE = {mse_val:.2f}, MAE = {mae_val:.2f}, R2 = {r2_val:.4f}")
print(f"  Test:  MSE = {mse_test:.2f}, MAE = {mae_test:.2f}, R2 = {r2_test:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(history_own['train_loss'], label='Train loss')
plt.plot(history_own['val_loss'], label='Val loss')
plt.xlabel('Эпоха')
plt.ylabel('MSE (scaled)')
plt.title('Динамика обучения (собственный MLP, регрессия)')
plt.legend()
plt.grid(True)
plt.show()


# Библиотечная реализация (sklearn)

print("\n--- Обучение библиотечного MLPRegressor (sklearn) ---")
mlp_reg_sk = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                          solver='adam', learning_rate_init=0.001,
                          max_iter=200, batch_size=32, random_state=42, verbose=False)
mlp_reg_sk.fit(X_train, y_train.ravel())

y_train_pred_sk = scaler_y.inverse_transform(mlp_reg_sk.predict(X_train).reshape(-1, 1))
y_val_pred_sk = scaler_y.inverse_transform(mlp_reg_sk.predict(X_val).reshape(-1, 1))
y_test_pred_sk = scaler_y.inverse_transform(mlp_reg_sk.predict(X_test).reshape(-1, 1))

mse_train_sk, mae_train_sk, r2_train_sk = calc_metrics(y_train_true, y_train_pred_sk)
mse_val_sk, mae_val_sk, r2_val_sk = calc_metrics(y_val_true, y_val_pred_sk)
mse_test_sk, mae_test_sk, r2_test_sk = calc_metrics(y_test_true, y_test_pred_sk)

print("\nРезультаты библиотечного MLPRegressor:")
print(f"  Train: MSE = {mse_train_sk:.2f}, MAE = {mae_train_sk:.2f}, R2 = {r2_train_sk:.4f}")
print(f"  Val:   MSE = {mse_val_sk:.2f}, MAE = {mae_val_sk:.2f}, R2 = {r2_val_sk:.4f}")
print(f"  Test:  MSE = {mse_test_sk:.2f}, MAE = {mae_test_sk:.2f}, R2 = {r2_test_sk:.4f}")


print("\n--- Сравнение метрик на тестовой выборке ---")
comparison = pd.DataFrame({
    'Модель': ['Собственная', 'sklearn'],
    'MSE': [mse_test, mse_test_sk],
    'MAE': [mae_test, mae_test_sk],
    'R2': [r2_test, r2_test_sk]
})
print(comparison.to_string(index=False))


# 2.2 МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН-КЛАССИФИКАТОР (Ожирение)


print("\n" + "=" * 70)
print("2.2 КЛАССИФИКАЦИЯ: УРОВЕНЬ ОЖИРЕНИЯ")
print("=" * 70)


df_obesity = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
print("\nПервые 5 строк:")
print(df_obesity.head())
print(f"\nРазмер датасета: {df_obesity.shape}")
print(f"Целевые классы: {df_obesity['NObeyesdad'].unique()}")


X_ob = df_obesity.drop('NObeyesdad', axis=1)
y_ob = df_obesity['NObeyesdad']


cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']


X_ob_encoded = pd.get_dummies(X_ob, columns=cat_cols, drop_first=False)


X_ob_encoded = X_ob_encoded.astype(float)


scaler_ob = StandardScaler()
X_ob_encoded[num_cols] = scaler_ob.fit_transform(X_ob_encoded[num_cols])


label_encoder = LabelEncoder()
y_ob_encoded = label_encoder.fit_transform(y_ob)
n_classes = len(label_encoder.classes_)
print(f"Классы: {list(label_encoder.classes_)} -> {list(range(n_classes))}")


y_ob_onehot = np.eye(n_classes)[y_ob_encoded]


X_train_ob, X_test_ob, y_train_ob, y_test_ob = train_test_split(
    X_ob_encoded.values.astype(float), y_ob_encoded, test_size=0.2, random_state=42, stratify=y_ob_encoded
)
_, _, y_train_ob_onehot, y_test_ob_onehot = train_test_split(
    X_ob_encoded.values.astype(float), y_ob_onehot, test_size=0.2, random_state=42, stratify=y_ob_encoded
)

X_train_ob, X_val_ob, y_train_ob, y_val_ob, y_train_ob_onehot, y_val_ob_onehot = train_test_split(
    X_train_ob, y_train_ob, y_train_ob_onehot, test_size=0.2, random_state=42, stratify=y_train_ob
)

print(f"\nРазмеры: X_train {X_train_ob.shape}, y_train {y_train_ob.shape}, X_val {X_val_ob.shape}, X_test {X_test_ob.shape}")


print(f"Тип X_train_ob: {X_train_ob.dtype}")
print(f"Тип X_val_ob: {X_val_ob.dtype}")
print(f"Тип X_test_ob: {X_test_ob.dtype}")



print("\n--- Обучение собственного MLP-классификатора ---")
mlp_clf_own = MLP(layers=[X_train_ob.shape[1], 64, 32, n_classes],
                  activations=['relu', 'relu', 'softmax'],
                  learning_rate=0.002, epochs=400, batch_size=32, random_state=42)
history_clf_own = mlp_clf_own.fit(X_train_ob, y_train_ob_onehot, loss_type='crossentropy',
                                   X_val=X_val_ob, y_val=y_val_ob_onehot, verbose=True)

y_train_pred_proba = mlp_clf_own.predict(X_train_ob)
y_val_pred_proba = mlp_clf_own.predict(X_val_ob)
y_test_pred_proba = mlp_clf_own.predict(X_test_ob)

y_train_pred = np.argmax(y_train_pred_proba, axis=1)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix_manual(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1(cm):
    prec = np.diag(cm) / (np.sum(cm, axis=0) + 1e-15)
    rec = np.diag(cm) / (np.sum(cm, axis=1) + 1e-15)
    f1 = 2 * prec * rec / (prec + rec + 1e-15)
    return np.mean(prec), np.mean(rec), np.mean(f1)

acc_train_own = accuracy(y_train_ob, y_train_pred)
acc_val_own = accuracy(y_val_ob, y_val_pred)
acc_test_own = accuracy(y_test_ob, y_test_pred)
cm_test_own = confusion_matrix_manual(y_test_ob, y_test_pred, n_classes)
prec_own, rec_own, f1_own = precision_recall_f1(cm_test_own)

print(f"\nРезультаты собственного MLP (классификация):")
print(f"  Train accuracy: {acc_train_own:.4f}")
print(f"  Val accuracy:   {acc_val_own:.4f}")
print(f"  Test accuracy:  {acc_test_own:.4f}")
print(f"  Macro Precision: {prec_own:.4f}, Recall: {rec_own:.4f}, F1: {f1_own:.4f}")
print("Confusion matrix (test):\n", cm_test_own)


plt.figure(figsize=(10,5))
plt.plot(history_clf_own['train_loss'], label='Train loss')
plt.plot(history_clf_own['val_loss'], label='Val loss')
plt.xlabel('Эпоха')
plt.ylabel('Cross-entropy loss')
plt.title('Динамика обучения (собственный MLP, классификация)')
plt.legend()
plt.grid(True)
plt.show()



print("\n--- Обучение библиотечного MLPClassifier (sklearn) ---")
mlp_clf_sk = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                           solver='adam', learning_rate_init=0.001,
                           max_iter=200, batch_size=32, random_state=42, verbose=False)
mlp_clf_sk.fit(X_train_ob, y_train_ob)

y_train_pred_sk = mlp_clf_sk.predict(X_train_ob)
y_val_pred_sk = mlp_clf_sk.predict(X_val_ob)
y_test_pred_sk = mlp_clf_sk.predict(X_test_ob)

acc_train_sk = accuracy_score(y_train_ob, y_train_pred_sk)
acc_val_sk = accuracy_score(y_val_ob, y_val_pred_sk)
acc_test_sk = accuracy_score(y_test_ob, y_test_pred_sk)
cm_test_sk = confusion_matrix(y_test_ob, y_test_pred_sk)
report = classification_report(y_test_ob, y_test_pred_sk, output_dict=True)
prec_sk = report['macro avg']['precision']
rec_sk = report['macro avg']['recall']

print(f"\nРезультаты библиотечного MLPClassifier:")
print(f"  Train accuracy: {acc_train_sk:.4f}")
print(f"  Val accuracy:   {acc_val_sk:.4f}")
print(f"  Test accuracy:  {acc_test_sk:.4f}")
print(f"  Macro Precision: {prec_sk:.4f}, Recall: {rec_sk:.4f}")
print("Confusion matrix (test):\n", cm_test_sk)


print("\n--- Сравнение метрик на тестовой выборке ---")
comparison_clf = pd.DataFrame({
    'Модель': ['Собственная', 'sklearn'],
    'Accuracy': [acc_test_own, acc_test_sk],
    'Precision (macro)': [prec_own, prec_sk],
    'Recall (macro)': [rec_own, rec_sk],
})
print(comparison_clf.to_string(index=False))


