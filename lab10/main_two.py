import numpy as np  # для роботи з масивами та математичних обчислень
from scipy.spatial import distance  # для обчислення відстаней між точками
import matplotlib.pyplot as plt  # для візуалізації
from sklearn.preprocessing import MinMaxScaler  # для нормалізації даних


class KohonenNetwork:
    def __init__(self, input_dim, num_clusters, learning_rate=0.1, epochs=200):
        self.input_dim = input_dim  # розмірність вхідних даних (у нашому випадку 2)
        self.num_clusters = num_clusters  # кількість кластерів
        self.learning_rate = learning_rate  # швидкість навчання
        self.epochs = epochs  # кількість епох навчання
        # Ініціалізація випадкових ваг для кожного кластера
        self.weights = np.random.rand(num_clusters, input_dim)

    def train(self, data):
        error_history = []  # для збереження історії помилок

        for epoch in range(self.epochs):
            total_error = 0
            np.random.shuffle(data)  # перемішування даних для кращого навчання

            for sample in data:
                # Знаходження найближчого нейрона (нейрона-переможця)
                distances = [distance.euclidean(sample, weight) for weight in self.weights]
                winner = np.argmin(distances)

                # Оновлення ваг нейрона-переможця
                self.weights[winner] += self.learning_rate * (sample - self.weights[winner])

                total_error += distances[winner]

            # Зменшення швидкості навчання з часом
            self.learning_rate *= 0.95

            # Збереження середньої помилки для епохи
            error_history.append(total_error / len(data))

        return error_history

    def predict(self, data):
        predictions = []
        for sample in data:
            # Знаходження найближчого кластера для кожної точки
            distances = [distance.euclidean(sample, weight) for weight in self.weights]
            predictions.append(np.argmin(distances))
        return np.array(predictions)


def main():
    # Завантаження даних
    data = np.array([
        [7.6, 8.7],
        [6.7, 8.1],
        [7.0, 10.0],
        [7.1, 7.1],
        [6.8, 9.0],
        [6.5, 7.5],
        [7.4, 9.1],
        [8.8, 7.5],
        [6.0, 8.7],
        [7.7, 8.4],
        [6.5, 11.5],
        [7.6, 7.8],
        [7.1, 10.4],
        [7.9, 6.2],
        [6.5, 9.0],
        [7.9, 8.8],
        [8.0, 9.6],
        [7.7, 7.9],
        [5.7, 9.4],
        [6.0, 8.8]
    ])

    # Нормалізація даних
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # Створення та навчання мережі
    network = KohonenNetwork(input_dim=2, num_clusters=3, learning_rate=0.1, epochs=200)
    error_history = network.train(data_normalized)

    # Отримання результатів кластеризації
    cluster_indices = network.predict(data_normalized)

    # Визначення центрів кластерів
    centers = np.zeros((network.num_clusters, network.input_dim))
    for i in range(network.num_clusters):
        cluster_points = data[cluster_indices == i]
        centers[i] = np.mean(cluster_points, axis=0) if len(cluster_points) > 0 else np.zeros(network.input_dim)

    # Підрахунок кількості елементів у кожному кластері
    cluster_sizes = [np.sum(cluster_indices == i) for i in range(network.num_clusters)]

    # Візуалізація результатів
    colors = ['b', 'r', 'g']
    plt.figure(figsize=(10, 8))

    # Візуалізація точок даних
    for i in range(network.num_clusters):
        cluster_points = data[cluster_indices == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        c=colors[i], label=f'Кластер {i + 1}')

    # Візуалізація центрів кластерів
    for i in range(network.num_clusters):
        plt.scatter(centers[i, 0], centers[i, 1], c=colors[i],
                    marker='x', s=200, linewidths=3,
                    label=f'Центр кластера {i + 1}')

    plt.title('Результати кластеризації мережею Кохонена')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.legend()

    # Візуалізація помилки навчання
    plt.figure(figsize=(10, 6))
    plt.plot(error_history)
    plt.title('Зміна помилки в процесі навчання')
    plt.xlabel('Епоха')
    plt.ylabel('Середня помилка')
    plt.grid(True)

    # Виведення результатів
    print('\nКількість елементів у кластерах:')
    for i in range(network.num_clusters):
        print(f'Кластер {i + 1}: {cluster_sizes[i]} елементів')

    print('\nКоординати центрів кластерів:')
    for i in range(network.num_clusters):
        print(f'Кластер {i + 1}: x1 = {centers[i, 0]:.2f}, x2 = {centers[i, 1]:.2f}')

    plt.show()


if __name__ == "__main__":
    main()