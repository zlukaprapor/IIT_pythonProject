import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import os


def load_and_preprocess_data(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    X = []
    y = []
    original_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            filepath = os.path.join(folder_path, filename)
            try:
                with Image.open(filepath) as img:
                    original_images.append(np.array(img))
                    if img.mode != 'L':
                        img = img.convert('L')
                    if img.size != (28, 28):
                        img = img.resize((28, 28))
                    X.append(np.array(img).reshape(-1))
                    y.append(filename[0])
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    if not X:
        raise ValueError("No valid TIF files found in the specified folder.")

    return np.array(X), np.array(y), original_images


def check_dimensions(X):
    print(f"Shape of X in check_dimensions: {X.shape}")
    if X.ndim != 2:
        print(f"Unexpected number of dimensions: {X.ndim}")
        return False
    if X.shape[1] != 784:  # 28 * 28 = 784
        print(f"Unexpected number of features: {X.shape[1]}")
        return False
    return True


def train_mlp(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp.fit(X_scaled, y)

    return mlp, scaler


def evaluate_model(mlp, scaler, X, y, images, title):
    X_scaled = scaler.transform(X)
    y_pred = mlp.predict(X_scaled)

    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"\n{title}")
    print(f"Загальна точність: {accuracy:.2f}")
    print("\nТочність розпізнавання кожного символу:")
    for i, char in enumerate(np.unique(y)):
        char_accuracy = conf_matrix[i, i] / np.sum(conf_matrix[i])
        print(f"{char}: {char_accuracy:.2f}")

    print("\nРезультати розпізнавання:")
    for i, (true_char, pred_char) in enumerate(zip(y, y_pred)):
        print(f"Зображення {i + 1}: Справжній символ: {true_char}, Розпізнаний символ: {pred_char}")

    visualize_results(images, y, y_pred, title)


def add_noise(image, noise_factor=10.0):
    noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    return np.clip(noisy_image, 0., 255.)


def visualize_results(images, y_true, y_pred, title):
    n = len(images)
    fig, axs = plt.subplots(2, n, figsize=(n * 2, 4))
    fig.suptitle(title)
    for i in range(n):
        axs[0, i].imshow(images[i], cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title(f"True: {y_true[i]}")

        axs[1, i].imshow(images[i], cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Pred: {y_pred[i]}")
    plt.tight_layout()
    plt.show()


# Основна частина програми
folder_path = r"mnt"  # Замініть це на реальний шлях до вашої папки з TIF-файлами

try:
    X, y, original_images = load_and_preprocess_data(folder_path)

    print(f"Loaded {len(X)} images")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # Візуалізація оригінальних зображень
    visualize_results(original_images, y, y, "Original Images")

    if check_dimensions(X):
        mlp, scaler = train_mlp(X, y)

        # Візуалізація препроцесованих зображень
        preprocessed_images = [x.reshape(28, 28) for x in X]
        visualize_results(preprocessed_images, y, y, "Preprocessed Images")

        # Оцінка на навчальному наборі
        evaluate_model(mlp, scaler, X, y, preprocessed_images, "Оцінка на навчальному наборі")

        # Тестування на зашумлених даних
        X_noisy = np.array([add_noise(x.reshape(28, 28)).flatten() for x in X])
        noisy_images = [x.reshape(28, 28) for x in X_noisy]

        # Візуалізація зашумлених зображень
        visualize_results(noisy_images, y, y, "Noisy Images")

        # Оцінка на зашумлених даних
        evaluate_model(mlp, scaler, X_noisy, y, noisy_images, "Оцінка на зашумлених даних")
    else:
        print("Перевірте розмірність вхідних даних перед продовженням.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback

    traceback.print_exc()