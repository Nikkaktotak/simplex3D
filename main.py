import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
import buildModelLib as ml
import time
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Константи
number_of_repeating = 10  # кількість запусків програми з цими ж даними
n = 0  # кількість точок всередині кулі
r = 1000  # радіус кулі
initial_point_counts = [500, 1000, 5000, 10000, 20000]  # початкова кількість точок
target_vertex_count_values = [100]  # кількість точок, до якої потрібно видаляти вершини
xml_file_path = "D:\\Article\\simplex3D\\statistics\\statistics2.xml"  # Шлях до XML файлу

# Створення кореневого елемента XML
root = ET.Element("statistics")

# Закоментовано: Запуск функції повного перебору для значень initial_point_counts 50 та 100
# full_times = {}
# for initial_point_count in [50, 100]:
#     if initial_point_count in initial_point_counts:
#         # Генерація точок
#         points, V = ml.generatePoints(n, initial_point_count, r)
#
#         # Запуск функції повного перебору на початковому наборі точок
#         start_full_time = time.time()
#         max_simplex_full, max_volume_full = ml.find_max_volume_simplex(points)
#         end_full_time = time.time()
#         full_time = end_full_time - start_full_time
#
#         # Зберігання часу повного перебору
#         full_times[initial_point_count] = full_time

for initial_point_count in initial_point_counts:
    initial_block = ET.SubElement(root, "initial_point_count_block", count=str(initial_point_count))

    for target_vertex_count in target_vertex_count_values:
        if target_vertex_count < initial_point_count:
            total_success_rate = 0
            success_rates = []

            print(f"Запуск програми для пари (початкова кількість точок: {initial_point_count}, цільова кількість точок: {target_vertex_count})")

            for i in tqdm(range(number_of_repeating), desc=f"Пара (initial_point_count={initial_point_count}, target_vertex_count={target_vertex_count})", leave=True):
                # Генерація точок
                points, V = ml.generatePoints(n, initial_point_count, r)

                # Створення об'єкта MyConvexHull і візуалізація початкової опуклої оболонки
                convex_hull = ml.MyConvexHull(points)

                # Початок відліку часу
                start_time = time.time()

                # Виконання видалення точок до тих пір, поки кількість точок у convex_hull.vertices не стане рівною target_vertex_count
                while len(convex_hull.vertices) > target_vertex_count:
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
                success_rates.append(success_rate)
                total_success_rate += success_rate

                # Додавання результатів до XML для кожного пробігу
                run_element = ET.SubElement(initial_block, "run", number=str(i+1))
                ET.SubElement(run_element, "target_vertex_count").text = str(target_vertex_count)
                ET.SubElement(run_element, "success_rate").text = f"{success_rate:.2f}"

                # Виведення результатів кожного пробігу
                print(f"Результати пробігу {i+1} (початкова кількість точок: {initial_point_count}, цільова кількість точок: {target_vertex_count}):")
                print("Відсоток наближеності значення об'єму:", success_rate, "%")
                print()

            # Обчислення середніх значень
            average_success_rate = round(total_success_rate / number_of_repeating, 2)
            best_success_rate = round(max(success_rates), 2)
            worst_success_rate = round(min(success_rates), 2)

            # Додавання середнього значення успішності до XML
            average_element = ET.SubElement(initial_block, "average")
            ET.SubElement(average_element, "target_vertex_count").text = str(target_vertex_count)
            ET.SubElement(average_element, "average_success_rate").text = f"{average_success_rate:.2f}"
            ET.SubElement(average_element, "best_success_rate").text = f"{best_success_rate:.2f}"
            ET.SubElement(average_element, "worst_success_rate").text = f"{worst_success_rate:.2f}"

            # Виведення середніх результатів
            print(f"\nСередні результати для пари (початкова кількість точок: {initial_point_count}, цільова кількість точок: {target_vertex_count}):")
            print("Середній відсоток наближеності значення об'єму:", average_success_rate, "%")
            print("Найкращий відсоток наближеності значення об'єму:", best_success_rate, "%")
            print("Найгірший відсоток наближеності значення об'єму:", worst_success_rate, "%")
            print()

# Запис результатів у XML файл з гарним форматуванням
tree = ET.ElementTree(root)
xml_str = ET.tostring(root, encoding='utf-8')
pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")

with open(xml_file_path, "w", encoding='utf-8') as f:
    f.write(pretty_xml_str)

print(f"Результати записані в {xml_file_path}")
