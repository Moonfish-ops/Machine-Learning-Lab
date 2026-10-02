import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data/regress_data2.csv")

X_raw = data.iloc[:, 0:2].to_numpy(dtype=float)
y = data.iloc[:, 2].to_numpy(dtype=float)

mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0)

X = (X_raw - mean) / std


m = len(y)

X = np.column_stack(( np.ones(m), X ))


def compute_loss(X, y, w):

    y_pred = X @ w
    error = y_pred - y

    return np.sum(error ** 2) / (2 * len(y))


def gradient_descent(X, y, learning_rate=0.01, epochs=1000):

    w = np.zeros(X.shape[1])

    loss_history = []

    for epoch in range(epochs):

        y_pred = X @ w

        error = y_pred - y

        gradient = X.T @ error / len(y)

        w = w - learning_rate * gradient

        loss = compute_loss(X, y, w)

        loss_history.append(loss)

    return w, loss_history



w, loss_history = gradient_descent(X, y)

print("模型参数 w =", w)
print("最终 loss =", loss_history[-1])



# 画 loss 曲线


plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")

plt.show()

# 画三维回归平面


fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection="3d")


ax.scatter(
    X_raw[:, 0],
    X_raw[:, 1],
    y
)


x1 = np.linspace(
    X_raw[:, 0].min(),
    X_raw[:, 0].max(),
    30
)

x2 = np.linspace(
    X_raw[:, 1].min(),
    X_raw[:, 1].max(),
    30
)

x1_grid, x2_grid = np.meshgrid(x1, x2)


x1_norm = (x1_grid - mean[0]) / std[0]
x2_norm = (x2_grid - mean[1]) / std[1]
y_grid = (
    w[0]
    + w[1] * x1_norm
    + w[2] * x2_norm
)


ax.plot_surface(
    x1_grid,
    x2_grid,
    y_grid,
    alpha=0.5
)


ax.set_xlabel("House Size")
ax.set_ylabel("Number of Rooms")
ax.set_zlabel("Price")
ax.set_title("Multiple Linear Regression")

plt.show()