import numpy as np
import matplotlib.pyplot as plt


# 构造实验数据
np.random.seed(42)

x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 100)


# 构造多项式特征
def polynomial_features(x, degree):

    X = np.ones((len(x), degree + 1))

    for i in range(1, degree + 1):
        X[:, i] = x ** i

    return X


# 特征标准化
def normalize(X):

    mean = np.mean(X[:, 1:], axis=0)
    std = np.std(X[:, 1:], axis=0)

    std[std == 0] = 1

    X[:, 1:] = (X[:, 1:] - mean) / std

    return X, mean, std


# 损失函数
def compute_loss(X, y, w):

    y_pred = X @ w
    error = y_pred - y

    return np.sum(error ** 2) / (2 * len(y))


# 梯度下降
def gradient_descent(X, y, learning_rate=0.01, epochs=20000):

    w = np.zeros(X.shape[1])

    for epoch in range(epochs):

        y_pred = X @ w

        error = y_pred - y

        gradient = X.T @ error / len(y)

        w = w - learning_rate * gradient

    return w


# 分别测试 M = 0, 1, 3, 9
degrees = [0, 1, 7, 149]

x_plot = np.linspace(0, 1, 300)

plt.figure(figsize=(10, 8))


for index, degree in enumerate(degrees):

    # 生成多项式特征
    X = polynomial_features(x, degree)

    if degree > 0:
        X, mean, std = normalize(X)

    # 训练
    w = gradient_descent(X, y)

    X_plot = polynomial_features(x_plot, degree)
    if degree > 0:
        X_plot[:, 1:] = (X_plot[:, 1:] - mean) / std

    y_plot = X_plot @ w


    plt.subplot(2, 2, index + 1)

    plt.scatter(x, y)

    plt.plot(
        x_plot,
        np.sin(2 * np.pi * x_plot),
        linestyle="--",
        label="True"
    )

    plt.plot(
        x_plot,
        y_plot,
        label=f"M={degree}"
    )

    plt.title(f"M = {degree}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()


plt.tight_layout()
plt.show()