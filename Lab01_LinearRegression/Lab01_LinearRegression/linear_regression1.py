import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data/regress_data1.csv")

x = data.iloc[:, 0].to_numpy()
y = data.iloc[:, 1].to_numpy()


m = len(x)

X = np.column_stack((np.ones(m), x))


def compute_loss(X, y, w):
    y_pred = X @ w
    error = y_pred - y

    return np.sum(error ** 2) / (2 * len(y))


def gradient_descent(X, y, learning_rate=0.01, epochs=1000):

    w = np.zeros(X.shape[1])

    for epoch in range(epochs):

        # 预测
        y_pred = X @ w

        # 误差
        error = y_pred - y

        # 梯度
        gradient = X.T @ error / len(y)

        # 更新参数
        w = w - learning_rate * gradient

    return w


w = gradient_descent(X, y)


y_pred = X @ w


index = np.argsort(x)

plt.scatter(x, y, label="Original Data")
plt.plot(x[index], y_pred[index], label="Regression Line")
plt.xlabel("Population")
plt.ylabel("Profit")
plt.legend()
plt.show()