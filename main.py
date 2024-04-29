# main.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
import buildModelLib as ml

# Генерація n випадкових точок в просторі
n = 100 #к-ть точок всередині кулі
m = 100 #к-ть точок на сфері
r = 1000
points = ml.generatePoints(n, m, r)

# Створення опуклого многокутника
hull = ConvexHull(points)

# Отримання граней та їх вершин
faces = hull.simplices
vertices = hull.points

# Визначення точок, які не утворюють грані
non_hull_points_mask = np.ones(n + m, dtype=bool)
non_hull_points_mask[faces.flatten()] = False
non_hull_points = np.where(non_hull_points_mask)[0]

# Генерація випадкових кольорів для граней
face_colors = np.random.rand(len(faces), 3)

# Створення фігури та 3D-вісей
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Малювання многокутника з різнокольоровими гранями
poly3d = Poly3DCollection(vertices[faces], facecolors=face_colors, linewidths=1, edgecolors='r', alpha=0.5)
ax.add_collection3d(poly3d)

# Малювання точок, які не утворюють грані (чорні)
ax.scatter(points[non_hull_points, 0], points[non_hull_points, 1], points[non_hull_points, 2],
           color='black', marker='o')

# Малювання точок, які утворюють грані (червоні)
ax.scatter(points[faces.flatten(), 0], points[faces.flatten(), 1], points[faces.flatten(), 2],
           color='red', marker='o')

# Налаштування відображення
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Convex Polyhedron in 3D with Non-Hull Points')

# Відображення графіку
plt.show()

import numpy as np


