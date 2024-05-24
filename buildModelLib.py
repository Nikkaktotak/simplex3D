import numpy as np
from scipy.spatial import ConvexHull as SciPyConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class MyConvexHull:
    def __init__(self, points):
        """
        Ініціалізація об'єкта MyConvexHull.
        Створює опуклу оболонку для заданих точок і візуалізує її.
        """
        self.points = points
        self.vertices = []
        self.edges = []
        self.faces = []
        self.deleted_vertices = []
        self._compute_hull()
        self.visualize()  # Побудова зображення після створення об'єкта

    def _compute_hull(self):
        """
        Обчислює опуклу оболонку для заданих точок, оновлює вершини, грані та ребра.
        """
        hull = SciPyConvexHull(self.points)
        self.vertices = hull.points.tolist()
        self.faces = hull.simplices.tolist()
        self.edges = self._compute_edges(self.faces)

    @staticmethod
    def _compute_edges(faces):
        """
        Визначає всі унікальні ребра на основі заданих граней.

        Parameters:
            faces (list): Список граней.

        Returns:
            list: Список унікальних ребер.
        """
        edges = set()
        for face in faces:
            start = face[0]
            for end in face[1:]:
                edges.add(tuple(sorted((start, end))))
                start = end
            edges.add(tuple(sorted((face[-1], face[0]))))
        return list(edges)

    def update(self, verticeForDelete):
        """
        Оновлює опуклу оболонку шляхом видалення вказаної вершини.

        Parameters:
            verticeForDelete (int): Індекс вершини, яку потрібно видалити.
        """
        points = np.array(self.vertices)

        # Знаходження сусідів вершини, яку потрібно видалити
        hull = SciPyConvexHull(points)
        neighbors = set()
        for simplex in hull.simplices:
            if verticeForDelete in simplex:
                neighbors.update(simplex)
        neighbors.remove(verticeForDelete)
        neighbors = list(neighbors)

        # Видалення вершини
        points = np.delete(points, verticeForDelete, axis=0)

        # Перебудова опуклої оболонки для сусідів
        sub_hull = SciPyConvexHull(points[neighbors])

        # Оновлення граней та ребер
        new_faces = sub_hull.simplices
        new_edges = set()
        for face in new_faces:
            start = face[0]
            for end in face[1:]:
                new_edges.add(tuple(sorted((start, end))))
                start = end
            new_edges.add(tuple(sorted((face[-1], face[0]))))

        # Конвертація множини в список
        new_edges = list(new_edges)

        # Мапування локальних індексів на глобальні індекси
        global_faces = []
        for face in new_faces:
            global_faces.append([neighbors[i] for i in face])

        global_edges = []
        for edge in new_edges:
            global_edges.append([neighbors[i] for i in edge])

        # Оновлення атрибутів об'єкта
        self.vertices = points.tolist()
        self.edges = global_edges
        self.faces = global_faces

        # Додавання видаленої вершини до списку
        self.deleted_vertices.append(verticeForDelete)

    def visualize(self):
        """
        Візуалізує поточну опуклу оболонку.
        """
        fig = plt.figure(figsize=(10, 8))  # Збільшений розмір фігури
        ax = fig.add_subplot(111, projection='3d')

        # Використання вершин та граней для побудови полігонів
        poly3d = [np.array(self.vertices)[face] for face in self.faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='pink', linewidths=1, edgecolors='k', alpha=0.25))

        # Відображення всіх вершин
        vertices_array = np.array(self.vertices)
        ax.scatter(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2], color='k')

        # Встановлення масштабу з відступами
        padding = 0.1  # 10% padding
        max_range = np.array([vertices_array[:, 0].max() - vertices_array[:, 0].min(),
                              vertices_array[:, 1].max() - vertices_array[:, 1].min(),
                              vertices_array[:, 2].max() - vertices_array[:, 2].min()]).max() / 2.0

        mid_x = (vertices_array[:, 0].max() + vertices_array[:, 0].min()) * 0.5
        mid_y = (vertices_array[:, 1].max() + vertices_array[:, 1].min()) * 0.5
        mid_z = (vertices_array[:, 2].max() + vertices_array[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range * (1 + padding), mid_x + max_range * (1 + padding))
        ax.set_ylim(mid_y - max_range * (1 + padding), mid_y + max_range * (1 + padding))
        ax.set_zlim(mid_z - max_range * (1 + padding), mid_z + max_range * (1 + padding))

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()


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

def find_point_for_delete(convex_hull):
    def edge_length(edge):
        return np.linalg.norm(np.array(convex_hull.vertices[edge[0]]) - np.array(convex_hull.vertices[edge[1]]))

    # 1. Знаходження найменшого ребра (по довжині)
    min_edge = min(convex_hull.edges, key=edge_length)

    # 2. Знаходження найменшого ребра серед ребер, що містять одну з точок min_edge[0] або min_edge[1]
    candidate_edges = [edge for edge in convex_hull.edges if min_edge[0] in edge or min_edge[1] in edge]
    candidate_edges.remove(min_edge)  # Видалення ребра min_edge з кандидатів

    min_edge_2 = min(candidate_edges, key=edge_length)

    # 3. Повернення індексу відповідної точки
    if min_edge[0] in min_edge_2:
        return min_edge[0]
    else:
        return min_edge[1]
