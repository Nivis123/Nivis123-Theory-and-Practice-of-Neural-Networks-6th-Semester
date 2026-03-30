import gzip
import numpy as np
import matplotlib.pyplot as plt
from time import time


def read_labels_from_file(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        nolab = int.from_bytes(f.read(4), 'big')
        labels = [int.from_bytes(f.read(1), 'big') for _ in range(nolab)]
    return labels

def read_images_from_file(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        noimg = int.from_bytes(f.read(4), 'big')
        nocol = int.from_bytes(f.read(4), 'big')
        norow = int.from_bytes(f.read(4), 'big')
        images = []
        for _ in range(noimg):
            img = []
            for _ in range(norow):
                row = []
                for _ in range(nocol):
                    row.append(int.from_bytes(f.read(1), 'big'))
                img.append(row)
            images.append(img)
    return np.array(images, dtype=np.float32)

print("Loading MNIST data...")
train_images = read_images_from_file('train-images-idx3-ubyte.gz')
train_labels = read_labels_from_file('train-labels-idx1-ubyte.gz')
test_images = read_images_from_file('t10k-images-idx3-ubyte.gz')
test_labels = read_labels_from_file('t10k-labels-idx1-ubyte.gz')

train_images /= 255.0
test_images /= 255.0


train_images = np.pad(train_images, ((0,0),(2,2),(2,2)), mode='constant')
test_images = np.pad(test_images, ((0,0),(2,2),(2,2)), mode='constant')


def one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

train_labels_onehot = one_hot(train_labels)
test_labels_onehot = one_hot(test_labels)


def im2col(images, kernel_h, kernel_w, stride, pad=0):
    """
    Convert image batch to column matrix for convolution.
    images: (n, c, h, w)
    Returns: cols (n*out_h*out_w, c*kernel_h*kernel_w), out_h, out_w
    """
    n, c, h, w = images.shape
    out_h = (h + 2*pad - kernel_h) // stride + 1
    out_w = (w + 2*pad - kernel_w) // stride + 1
    if pad > 0:
        images = np.pad(images, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((n, c, kernel_h, kernel_w, out_h, out_w))
    for i in range(kernel_h):
        i_max = i + stride*out_h
        for j in range(kernel_w):
            j_max = j + stride*out_w
            cols[:, :, i, j, :, :] = images[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.transpose(0,4,5,1,2,3).reshape(n*out_h*out_w, -1)
    return cols, out_h, out_w

def col2im(cols, shape, kernel_h, kernel_w, stride, pad=0):
    """
    Inverse of im2col.
    cols: (n*out_h*out_w, c*kernel_h*kernel_w)
    shape: (n, c, h, w) original input shape
    """
    n, c, h, w = shape
    out_h = (h + 2*pad - kernel_h) // stride + 1
    out_w = (w + 2*pad - kernel_w) // stride + 1
    cols = cols.reshape(n, out_h, out_w, c, kernel_h, kernel_w).transpose(0,3,4,5,1,2)
    if pad > 0:
        h_pad, w_pad = h+2*pad, w+2*pad
    else:
        h_pad, w_pad = h, w
    grad = np.zeros((n, c, h_pad, w_pad))
    for i in range(kernel_h):
        i_max = i + stride*out_h
        for j in range(kernel_w):
            j_max = j + stride*out_w
            grad[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    if pad > 0:
        grad = grad[:, :, pad:-pad, pad:-pad]
    return grad

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Xavier initialization (for tanh)
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (fan_in + fan_out))
        self.b = np.zeros(out_channels)
        self.dW = None
        self.db = None
        self.x = None
        self.cols = None
        self.out_h = None
        self.out_w = None

    def forward(self, x):
        self.x = x
        n, c, h, w = x.shape
        self.cols, self.out_h, self.out_w = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)
        W_col = self.W.reshape(self.out_channels, -1)
        out = self.cols.dot(W_col.T) + self.b
        out = out.reshape(n, self.out_h, self.out_w, self.out_channels).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        n, c, out_h, out_w = dout.shape
        batch_size = n
        dout_flat = dout.transpose(0,2,3,1).reshape(-1, self.out_channels)
        # Average gradients over batch
        self.dW = dout_flat.T.dot(self.cols).reshape(self.W.shape) / batch_size
        self.db = np.sum(dout_flat, axis=0) / batch_size
        dcols = dout_flat.dot(self.W.reshape(self.out_channels, -1))
        dx = col2im(dcols, self.x.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        return dx

class AvgPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.in_shape = None

    def forward(self, x):
        self.x = x
        self.in_shape = x.shape
        n, c, h, w = x.shape
        out_h = h // self.stride
        out_w = w // self.stride
        out = np.zeros((n, c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                out[:, :, i, j] = np.mean(x[:, :, i*self.stride:i*self.stride+self.pool_size,
                                            j*self.stride:j*self.stride+self.pool_size], axis=(2,3))
        return out

    def backward(self, dout):
        n, c, out_h, out_w = dout.shape
        dx = np.zeros(self.in_shape)
        for i in range(out_h):
            for j in range(out_w):
                dx[:, :, i*self.stride:i*self.stride+self.pool_size,
                      j*self.stride:j*self.stride+self.pool_size] += dout[:, :, i, j][:, :, None, None] / (self.pool_size**2)
        return dx

class Dense:
    def __init__(self, in_features, out_features):
        # Xavier initialization
        fan_in = in_features
        fan_out = out_features
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (fan_in + fan_out))
        self.b = np.zeros(out_features)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x.dot(self.W) + self.b

    def backward(self, dout):
        batch_size = dout.shape[0]
        self.dW = self.x.T.dot(dout) / batch_size
        self.db = np.sum(dout, axis=0) / batch_size
        dx = dout.dot(self.W.T)
        return dx

class Activation:
    def __init__(self, act_func):
        self.act_func = act_func
        self.x = None

    def forward(self, x):
        self.x = x
        return self.act_func(x)

    def backward(self, dout):
        if self.act_func == tanh:
            return dout * tanh_derivative(self.x)
        else:
            raise NotImplementedError

class Flatten:
    def forward(self, x):
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.in_shape)


class LeNet5:
    def __init__(self):
        self.conv1 = Conv2D(1, 6, kernel_size=5, stride=1, padding=0)   # 32->28
        self.act1 = Activation(tanh)
        self.pool1 = AvgPool2D(pool_size=2, stride=2)                    # 28->14
        self.conv2 = Conv2D(6, 16, kernel_size=5, stride=1, padding=0)   # 14->10
        self.act2 = Activation(tanh)
        self.pool2 = AvgPool2D(pool_size=2, stride=2)                    # 10->5
        self.flatten = Flatten()
        self.fc1 = Dense(16*5*5, 120)
        self.act3 = Activation(tanh)
        self.fc2 = Dense(120, 84)
        self.act4 = Activation(tanh)
        self.fc3 = Dense(84, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.act1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.act2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.act3.forward(x)
        x = self.fc2.forward(x)
        x = self.act4.forward(x)
        x = self.fc3.forward(x)
        return softmax(x)

    def backward(self, dout):
        dout = self.fc3.backward(dout)
        dout = self.act4.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.act3.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.pool2.backward(dout)
        dout = self.act2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.act1.backward(dout)
        dout = self.conv1.backward(dout)

    def update_params(self, lr):
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db


def train(model, X_train, y_train, X_test, y_test, epochs, batch_size, lr):
    n = X_train.shape[0]
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n)
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0
        for i in range(0, n, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_pred = model.forward(X_batch)
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_loss += loss

            dout = y_pred - y_batch
            model.backward(dout)

            model.update_params(lr)

        avg_loss = epoch_loss / (n // batch_size)
        train_losses.append(avg_loss)

        y_pred_test = model.forward(X_test)
        test_acc = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - test_acc: {test_acc:.4f}")

    return train_losses, test_accuracies


X_train = train_images[:, np.newaxis, :, :]
X_test = test_images[:, np.newaxis, :, :]
y_train = train_labels_onehot
y_test = test_labels_onehot

model = LeNet5()
start = time()
train_losses, test_accuracies = train(model, X_train, y_train, X_test, y_test,
                                      epochs=10, batch_size=64, lr=0.01)
print(f"Training time: {time()-start:.2f} seconds")

# Plot results
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy')

plt.subplot(1,2,2)
plt.plot(test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('manual_training.png')
plt.show()