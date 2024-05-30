import buildModelLib as ml
import time
from keras.utils import Progbar
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os


number_of_repeating = 10  # кількість запусків програми з цими ж даними
n = 0  # кількість точок всередині кулі
r = 100  # радіус кулі
initial_point_counts = [500, 1000, 2000, 5000, 10000, 15000]  # початкова кількість точок
target_vertex_count_values = [100]  # кількість точок, до якої потрібно видаляти вершини
xml_file_path = os.path.join(os.path.dirname(__file__), "statistics/statistics.xml")  # Шлях до XML файлу

# Створення кореневого елемента XML
root = ET.Element("statistics")

for initial_point_count in initial_point_counts:
    initial_block = ET.SubElement(root, "initial_point_count_block", count=str(initial_point_count))

    for target_vertex_count in target_vertex_count_values:
        if target_vertex_count < initial_point_count:
            total_success_rate = 0
            success_rates = []

            print(f"\nЗапуск програми для пари (початкова кількість точок: {initial_point_count}, цільова кількість точок: {target_vertex_count})")
            progbar = Progbar(target=number_of_repeating)

            for i in range(number_of_repeating):
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

                # Оновлення прогрес-бару
                progbar.update(i + 1)

            # Обчислення середніх значень
            average_success_rate = round(total_success_rate / number_of_repeating, 2)
            best_success_rate = round(max(success_rates), 2)
            worst_success_rate = round(min(success_rates), 2)

            # Додавання блоку з усіма значеннями успішності до XML
            all_success_rates_block = ET.SubElement(initial_block, "all_success_rates", target_vertex_count=str(target_vertex_count))
            for i, rate in enumerate(success_rates, start=1):
                run_element = ET.SubElement(all_success_rates_block, "run", number=str(i))
                ET.SubElement(run_element, "success_rate").text = f"{rate:.2f}"

            # Додавання середнього значення успішності до XML
            average_element = ET.SubElement(initial_block, "average")
            ET.SubElement(average_element, "target_vertex_count").text = str(target_vertex_count)
            ET.SubElement(average_element, "average_success_rate").text = f"{average_success_rate:.2f}"
            ET.SubElement(average_element, "best_success_rate").text = f"{best_success_rate:.2f}"
            ET.SubElement(average_element, "worst_success_rate").text = f"{worst_success_rate:.2f}"

            # Виведення середніх результатів
            print(f"Середні результати для пари (початкова кількість точок: {initial_point_count}, цільова кількість точок: {target_vertex_count}):")
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
