import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 读取数据
data = pd.read_csv("data/regress_data1.csv")

x = data.iloc[:, 0].to_numpy()
y = data.iloc[:, 1].to_numpy()


# 构造多项式特征
def polynomial_features(x, n):

    X = np.ones((len(x), n + 1))

    for i in range(1, n + 1):
        X[:, i] = x ** i

    return X


# 标准化
def normalize(X):

    # n=0 时只有偏置项，不需要标准化
    if X.shape[1] == 1:
        return X, None, None

    mean = np.mean(X[:, 1:], axis=0)
    std = np.std(X[:, 1:], axis=0)

    X[:, 1:] = (X[:, 1:] - mean) / std

    return X, mean, std


# 梯度下降
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):

    w = np.zeros(X.shape[1])

    for epoch in range(epochs):

        y_pred = X @ w

        error = y_pred - y

        gradient = X.T @ error / len(y)

        w = w - learning_rate * gradient

    return w


# 测试不同的 n
n_list = [0, 1, 9]

x_plot = np.linspace(x.min(), x.max(), 300)

plt.figure(figsize=(12, 4))


for index, n in enumerate(n_list):

    # 构造训练数据的多项式特征
    X = polynomial_features(x, n)

    # 标准化
    X, mean, std = normalize(X)

    # 梯度下降
    w = gradient_descent(X, y)

    # 构造画图用的数据
    X_plot = polynomial_features(x_plot, n)

    # 使用训练数据的 mean 和 std 进行标准化
    if n > 0:
        X_plot[:, 1:] = (X_plot[:, 1:] - mean) / std

    # 预测
    y_plot = X_plot @ w

    # 画图
    plt.subplot(1, 3, index + 1)

    plt.scatter(x, y)
    plt.plot(x_plot, y_plot)

    plt.title("n = " + str(n))
    plt.xlabel("Population")
    plt.ylabel("Profit")


plt.tight_layout()
plt.show()