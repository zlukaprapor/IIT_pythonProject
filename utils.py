# utils.py

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """Завантаження даних з файлу"""
    return np.loadtxt(filename)


def plot_data_and_weights(data, coordinates, weights, title):
    """Візуалізація даних та ваг"""
    plt.figure(figsize=(10, 8))

    # Побудова вхідних даних
    plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.5, label='Input data')

    # Побудова ваг нейронів
    plt.scatter(weights[:, 0], weights[:, 1], c='red', marker='x',
                s=200, label='Neurons')

    # З'єднання сусідніх нейронів лініями
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            if (abs(coordinates[i][0] - coordinates[j][0]) <= 1 and
                    abs(coordinates[i][1] - coordinates[j][1]) <= 1):
                plt.plot([weights[i][0], weights[j][0]],
                         [weights[i][1], weights[j][1]], 'gray', alpha=0.3)

    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt


def save_weights(coordinates, weights, filename):
    """Збереження ваг у файл"""
    with open(filename, 'w') as f:
        for coord, w in zip(coordinates, weights):
            f.write(f"({coord[0]}, {coord[1]}) {w[0]} {w[1]}\n")