# main.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
import buildModelLib as ml

# Генерація n випадкових точок в просторі
n = 40  # кількість точок всередині кулі
m = 100  # кількість точок на сфері
r = 1000  # радіус кулі

# Генерація точок
points, V = ml.generatePoints(n, m, r)

# Створення об'єкта MyConvexHull і візуалізація початкової опуклої оболонки
convex_hull = ml.MyConvexHull(points)


# Знаходження індексу точки для видалення
pointForDelete = ml.find_point_for_delete(convex_hull)

# Оновлення опуклої оболонки шляхом видалення знайденої точки
convex_hull.update(pointForDelete)


# Виведення кількості видалених точок у термінал
print(len(convex_hull.deleted_vertices))
