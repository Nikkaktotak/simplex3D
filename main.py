# main.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
import buildModelLib as ml
import time

# Константи
number_of_repeating = 10  # кількість запусків програми з цими ж даними
n = 0  # кількість точок всередині кулі
m = 100  # кількість точок на сфері
r = 1000  # радіус кулі

# Для зберігання результатів
total_time = 0
total_success_rate = 0

for i in range(number_of_repeating):
    # Генерація точок
    points, V = ml.generatePoints(n, m, r)

    # Створення об'єкта MyConvexHull і візуалізація початкової опуклої оболонки
    convex_hull = ml.MyConvexHull(points)

    # Початок відліку часу
    start_time = time.time()

    # Виконання видалення точок до тих пір, поки кількість точок у convex_hull.vertices не стане рівною 40
    while len(convex_hull.vertices) > 40:
        # Знаходження індексу точки для видалення
        pointForDelete = ml.find_point_for_delete(convex_hull)

        # Оновлення опуклої оболонки шляхом видалення знайденої точки
        convex_hull.update(pointForDelete)

    # Знаходження симплексу з максимальним об'ємом серед залишених точок
    max_simplex, max_volume = ml.find_max_volume_simplex(convex_hull.vertices)

    # Кінець відліку часу
    end_time = time.time()

    # Розрахунок і виведення відсотка успішності
    success_rate = (max_volume / V) * 100

    # Сумування результатів
    total_time += end_time - start_time
    total_success_rate += success_rate

    # Виведення проміжних результатів
    print(f"Запуск {i+1}:")
    print("Максимальний об'єм порахований знаючи точки що утворюють симплекс максимального обєму:", V)
    print("Максимальний об'єм порахований після видалення точок за допомогою досліджуваної евристики:", max_volume)
    print("Відсоток успішності:", success_rate, "%")
    print("Час виконання програми:", end_time - start_time, "секунд")
    print()

# Виведення середніх результатів
average_time = total_time / number_of_repeating
average_success_rate = total_success_rate / number_of_repeating

print("Середній час виконання одного пробігу програми:", average_time, "секунд")
print("Середній відсоток успішності:", average_success_rate, "%")
