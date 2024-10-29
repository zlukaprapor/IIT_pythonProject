# main.py

from kohonen_som import KohonenSOM
from utils import load_data, plot_data_and_weights, save_weights


def run_task(output_size_x, output_size_y, data_file, task_name):
    # Завантаження даних
    data = load_data(data_file)

    # Створення та навчання мережі
    som = KohonenSOM(
        output_size_x=output_size_x,
        output_size_y=output_size_y,
        initial_lr=0.1,
        lr_decay=0.01,
        initial_nbd_width=1.0,
        nbd_width_decay=0.01
    )

    # Навчання мережі
    som.train(data)

    # Отримання результатів
    coordinates, weights = som.get_weights_positions()

    # Візуалізація результатів
    plt = plot_data_and_weights(
        data, coordinates, weights,
        f'Task {task_name}: {output_size_x}x{output_size_y} map'
    )

    # Збереження графіка
    plt.savefig(f'task_{task_name}_plot.png')
    plt.close()

    # Збереження ваг
    save_weights(coordinates, weights, f'task_{task_name}_weights.txt')


if __name__ == "__main__":
    # Завдання 1: карта 4x2
    run_task(4, 2, 'data24.txt', '1')

    # Завдання 2: карта 4x4
    run_task(4, 4, 'data24.txt', '2')

    # Завдання 3: карта 2x3
    run_task(2, 3, 'data24.txt', '3')

    # Додаткове завдання: карта 4x2 з іншими даними
    run_task(4, 2, 'data33.txt', 'additional')