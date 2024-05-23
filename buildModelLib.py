import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull

def generatePoints(n, m, r, scale_x=2, scale_y=3, scale_z=1):
    # Генерація n випадкових точок всередині кулі радіусом r
    radii = r * np.random.rand(n) ** (1 / 3)
    phi = np.random.uniform(0, np.pi, n)
    theta = np.random.uniform(0, 2 * np.pi, n)

    x = radii * np.sin(phi) * np.cos(theta)
    y = radii * np.sin(phi) * np.sin(theta)
    z = radii * np.cos(phi)
    internal_points = np.column_stack((x, y, z))

    # Генерація m точок на поверхні сфери радіусом r
    m = m - 4
    phi_sphere = np.random.uniform(0, np.pi, m)
    theta_sphere = np.random.uniform(0, 2 * np.pi, m)

    x_sphere = r * np.sin(phi_sphere) * np.cos(theta_sphere)
    y_sphere = r * np.sin(phi_sphere) * np.sin(theta_sphere)
    z_sphere = r * np.cos(phi_sphere)
    sphere_points = np.column_stack((x_sphere, y_sphere, z_sphere))

    # Додавання чотирьох точок, які утворюють піраміду
    pyramid_points = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    pyramid_points = r * pyramid_points / np.linalg.norm(pyramid_points, axis=1)[:, np.newaxis]

    # Об'єднання всіх точок перед перетворенням
    all_points = np.vstack((internal_points, sphere_points, pyramid_points))

    # Застосування афінного перетворення до всіх точок
    transformed_points = applyAffineTransformToEllipse(all_points, scale_x, scale_y, scale_z)

    # Оновлення трансформованих точок тетраедра
    transformed_pyramid_points = transformed_points[-4:]  # Останні чотири точки

    # Обчислення об'єму тетраедра
    a = np.linalg.norm(transformed_pyramid_points[1] - transformed_pyramid_points[0])
    b = np.linalg.norm(transformed_pyramid_points[2] - transformed_pyramid_points[0])
    c = np.linalg.norm(transformed_pyramid_points[3] - transformed_pyramid_points[0])
    V = (a * b * c * np.sqrt(2)) / 12

    return transformed_points, V

def createConvexHull(points):
    # Створення опуклого многокутника
    hull = ConvexHull(points)

    # Отримання граней та їх вершин
    faces = hull.simplices
    vertices = hull.points
    # Визначення усіх унікальних ребер
    edges = set()

    for face in faces:
        # Сортування індексів, щоб (i, j) і (j, i) були тим самим ребром
        start = face[0]
        for end in face[1:]:
            edges.add(tuple(sorted((start, end))))
            start = end
        # Замкнути цикл ребер
        edges.add(tuple(sorted((face[-1], face[0]))))

    # Конвертувати множину назад у список для можливого використання
    edges = list(edges)

    # Визначення точок, які не утворюють грані
    non_hull_points_mask = np.ones(len(points), dtype=bool)
    non_hull_points_mask[faces.flatten()] = False
    non_hull_points = np.where(non_hull_points_mask)[0]

    # Генерація випадкових кольорів для граней
    face_colors = np.random.rand(len(faces), 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hull = ConvexHull(points)
    faces = hull.simplices
    vertices = hull.points

    # Малювання опуклого многокутника
    poly3d = [vertices[face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.25))

    # Малювання точок на графіку
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b')

    # Задання однакового масштабу для всіх вісей
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

    return vertices, edges, faces

def applyAffineTransformToEllipse(vertices, scale_x=2, scale_y=3, scale_z=1):
    """
    Застосовує афінне перетворення до набору вершин для перетворення кулі в еліпсоїд.

    Параметри:
        vertices (np.ndarray): Масив вершин (N x D), де N - кількість вершин, D - розмірність.
        scale_x (float): Коефіцієнт масштабування для осі X.
        scale_y (float): Коефіцієнт масштабування для осі Y.
        scale_z (float): Коефіцієнт масштабування для осі Z.

    Повертає:
        np.ndarray: Перетворені вершини.
    """
    # Матриця масштабування для перетворення кулі в еліпсоїд
    transform_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, scale_z]
    ])

    # Додаємо один вектор для обробки гомогенних координат
    homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])

    # Розширення матриці перетворення для включення гомогенної координати
    full_transform_matrix = np.eye(transform_matrix.shape[0] + 1)
    full_transform_matrix[:-1, :-1] = transform_matrix

    # Виконання матричного множення для перетворення
    transformed_vertices = homogeneous_vertices.dot(full_transform_matrix)

    # Видалення гомогенної координати
    return transformed_vertices[:, :-1]
