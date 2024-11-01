# kohonen_som.py

import numpy as np


class KohonenSOM:
    def __init__(self, output_size_x, output_size_y, input_dim=2,
                 initial_lr=0.1, lr_decay=0.01,
                 initial_nbd_width=1.0, nbd_width_decay=0.01):
        self.output_size_x = output_size_x
        self.output_size_y = output_size_y
        self.input_dim = input_dim

        # Ініціалізація ваг випадковими значеннями від -0.1 до 0.1
        self.weights = np.random.uniform(-0.1, 0.1,
                                         (output_size_x * output_size_y, input_dim))

        # Параметри навчання
        self.learning_rate = initial_lr
        self.lr_decay = lr_decay
        self.nbd_width = initial_nbd_width
        self.nbd_width_decay = nbd_width_decay

        # Створення координатної сітки для нейронів
        self.neuron_coordinates = np.array([(x, y)
                                            for x in range(output_size_x)
                                            for y in range(output_size_y)])

    def find_winner(self, x):
        """Знаходження нейрона-переможця"""
        distances = np.sqrt(np.sum((self.weights - x) ** 2, axis=1))
        return np.argmin(distances)

    def get_neighborhood(self, winner_idx, input_vector):
        """Розрахунок коефіцієнтів сусідства для всіх нейронів"""
        winner_coord = self.neuron_coordinates[winner_idx]

        # Обчислення евклідових відстаней до переможця на карті
        distances = np.sqrt(np.sum((self.neuron_coordinates - winner_coord) ** 2, axis=1))

        # Розрахунок коефіцієнтів сусідства
        h = np.exp(-distances ** 2 / (2 * self.nbd_width ** 2))

        return h

    def train_epoch(self, data):
        """Навчання мережі протягом однієї епохи"""
        # Перемішування даних
        np.random.shuffle(data)

        for x in data:
            # Знаходження переможця
            winner_idx = self.find_winner(x)

            # Отримання коефіцієнтів сусідства
            h = self.get_neighborhood(winner_idx, x)

            # Оновлення ваг
            for j in range(len(self.weights)):
                self.weights[j] += self.learning_rate * h[j] * (x - self.weights[j])

        # Оновлення параметрів навчання
        self.learning_rate *= (1 - self.lr_decay)
        self.nbd_width *= (1 - self.nbd_width_decay)

    def train(self, data, epochs=5000):
        """Повне навчання мережі"""
        for epoch in range(epochs):
            self.train_epoch(data)

    def get_weights_positions(self):
        """Повертає ваги та їх позиції на карті"""
        return self.neuron_coordinates, self.weights