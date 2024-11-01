import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms


class CounterPropagationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CounterPropagationNetwork, self).__init__()

        # Шар Кохонена
        self.kohonen = nn.Linear(input_size, hidden_size, bias=False)
        # Шар Гроссберга
        self.grossberg = nn.Linear(hidden_size, num_classes, bias=False)

        # Ініціалізація ваг
        nn.init.uniform_(self.kohonen.weight, -1, 1)
        nn.init.uniform_(self.grossberg.weight, -1, 1)

        self.hidden_size = hidden_size

    def forward(self, x):
        # Нормалізація вхідного вектора
        x = x / torch.norm(x, dim=1, keepdim=True)

        # Шар Кохонена
        kohonen_out = self.kohonen(x)

        # Winner-takes-all
        _, indices = torch.max(kohonen_out, dim=1)
        kohonen_activated = torch.zeros_like(kohonen_out)
        kohonen_activated[range(kohonen_out.size(0)), indices] = 1

        # Шар Гроссберга
        out = self.grossberg(kohonen_activated)
        return out


class TIFDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

        # Визначаємо перетворення для зображень
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Конвертація в відтінки сірого
            transforms.ToTensor(),  # Перетворення в тензор
            transforms.Normalize(mean=[0.5], std=[0.5])  # Нормалізація
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Завантаження та перетворення зображення
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)

        # Перетворення зображення в одновимірний вектор
        image = image.view(-1)

        return image, self.labels[idx]

def add_noise(image, noise_level=0.1):
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return noisy_image.clamp(0, 1)


def load_and_check_images(image_paths):
    """Завантаження та перевірка розмірності зображень"""
    images = []
    first_size = None

    for path in image_paths:
        try:
            image = Image.open(path)
            if first_size is None:
                first_size = image.size
            elif image.size != first_size:
                raise ValueError(
                    f"Розмір зображення {path} ({image.size}) відрізняється від першого зображення {first_size}")
            images.append(np.array(image))
        except Exception as e:
            raise Exception(f"Помилка при завантаженні {path}: {str(e)}")

    return first_size, images


def evaluate_per_class(model, test_loader, device, num_classes):
    """Оцінка точності для кожного класу"""
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()  # Ensure it's an integer
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Розрахунок точності для кожного класу
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Точність для класу {i}: {accuracy:.2f}%')

    # Загальна точність
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    total_accuracy = 100 * total_correct / total_samples
    print(f'Загальна точність: {total_accuracy:.2f}%')

    return total_accuracy


def main():
    # Параметри
    image_paths = ['1.tif', '3.tif', '7.tif']
    num_classes = 3
    hidden_size = 15  # Збільшено для кращої класифікації трьох класів
    num_epochs = 10000
    batch_size = 4
    learning_rate = 0.001

    # Перевірка наявності GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Використовується пристрій: {device}")

    try:
        # Завантаження та перевірка зображень
        image_size, _ = load_and_check_images(image_paths)
        print(f"Розмір зображень: {image_size}")
        input_size = image_size[0] * image_size[1]  # Загальна кількість пікселів

        # Створення міток (по одному зображенню для кожного класу)
        labels = list(range(num_classes))

        # Створення датасету та завантажувача
        dataset = TIFDataset(image_paths, labels)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Створення моделі
        model = CounterPropagationNetwork(input_size, hidden_size, num_classes).to(device)

        # Визначення функції втрат та оптимізатора
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Навчання
        print("\nПочаток навчання...")
        losses = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Епоха [{epoch + 1}/{num_epochs}], Втрати: {avg_loss:.4f}')

        # Оцінка результатів
        print("\nРезультати навчання:")
        evaluate_per_class(model, train_loader, device, num_classes)

        # Візуалізація процесу навчання
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Графік втрат під час навчання')
        plt.xlabel('Епоха')
        plt.ylabel('Втрати')
        plt.grid(True)
        plt.show()

        # Збереження моделі
        torch.save(model.state_dict(), 'counter_propagation_model.pth')
        print("\nМодель збережено в файл 'counter_propagation_model.pth'")

    except Exception as e:
        print(f"Помилка: {str(e)}")


if __name__ == "__main__":
    main()
