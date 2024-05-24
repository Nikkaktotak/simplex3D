import numpy as np
from scipy.spatial import ConvexHull as SciPyConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools


class MyConvexHull:
    def __init__(self, points):
        self.points = points
        self.vertices = []
        self.edges = []
        self.faces = []
        self.deleted_vertices = []
        self._compute_hull()
        #self.visualize()  # Побудова зображення після створення об'єкта

    def _compute_hull(self):
        hull = SciPyConvexHull(self.points)
        self.vertices = hull.points.tolist()
        self.faces = hull.simplices.tolist()
        self.edges = self._compute_edges(self.faces)

    @staticmethod
    def _compute_edges(faces):
        edges = set()
        for face in faces:
            start = face[0]
            for end in face[1:]:
                edges.add(tuple(sorted((start, end))))
                start = end
            edges.add(tuple(sorted((face[-1], face[0]))))
        return list(edges)

    def update(self, verticeForDelete):
        points = np.array(self.vertices)

        # Find the neighbors of the vertex to be deleted
        hull = SciPyConvexHull(points)
        neighbors = set()
        for simplex in hull.simplices:
            if verticeForDelete in simplex:
                neighbors.update(simplex)
        neighbors.remove(verticeForDelete)
        neighbors = list(neighbors)

        # Remove the vertex
        points = np.delete(points, verticeForDelete, axis=0)

        # Update neighbor indices
        neighbors = [i if i < verticeForDelete else i - 1 for i in neighbors]

        # Check if there are enough points to form a new simplex
        if len(points) < 4:
            return

        # Recompute the convex hull for the neighborhood
        sub_hull = SciPyConvexHull(points)

        # Update faces and edges
        new_faces = sub_hull.simplices
        new_edges = set()
        for face in new_faces:
            start = face[0]
            for end in face[1:]:
                new_edges.add(tuple(sorted((start, end))))
                start = end
            new_edges.add(tuple(sorted((face[-1], face[0]))))

        # Convert set to list
        new_edges = list(new_edges)

        # Update the attributes directly
        self.vertices = points.tolist()
        self.edges = new_edges
        self.faces = new_faces

        # Add the deleted vertex to the list
        self.deleted_vertices.append(verticeForDelete)

    def visualize(self):
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
    """
    Генерує n випадкових точок всередині кулі та m точок на поверхні сфери радіусом r.
    Додає правильний тетраедр, вершини якого лежать на сфері, та застосовує афінне перетворення для отримання еліпсоїда.

    Параметри:
        n (int): Кількість точок всередині кулі.
        m (int): Кількість точок на поверхні сфери.
        r (float): Радіус кулі.
        scale_x (float): Коефіцієнт масштабування для осі X.
        scale_y (float): Коефіцієнт масштабування для осі Y.
        scale_z (float): Коефіцієнт масштабування для осі Z.

    Повертає:
        np.ndarray: Перетворені вершини.
    """
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

    # Додавання точок, які утворюють правильний тетраедр
    sqrt2 = np.sqrt(2)
    pyramid_points = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    pyramid_points = r * pyramid_points / sqrt2

    # Об'єднання всіх точок перед перетворенням
    all_points = np.vstack((internal_points, sphere_points, pyramid_points))

    # Застосування афінного перетворення до всіх точок
    transformed_points = applyAffineTransformToEllipse(all_points, scale_x, scale_y, scale_z)

    # Отримання точок тетраедра після афінного перетворення
    transformed_pyramid_points = transformed_points[-4:]

    # Обчислення об'єму тетраедра
    V = calculate_tetrahedron_volume(transformed_pyramid_points)

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


def calculate_tetrahedron_volume(points):
    """
    Обчислює об'єм тетраедра, заданого чотирма точками.

    Parameters:
        points (ndarray): Масив розмірності (4, 3), що містить координати чотирьох точок.

    Returns:
        float: Об'єм тетраедра.
    """
    a = points[0]
    b = points[1]
    c = points[2]
    d = points[3]
    volume = np.abs(np.dot(np.cross(b - a, c - a), d - a)) / 6.0
    return volume


def find_max_volume_simplex(points):
    """
    Знаходить симплекс з максимальним об'ємом серед усіх можливих комбінацій чотирьох точок.

    Parameters:
        points (ndarray): Масив розмірності (n, 3), що містить координати точок.

    Returns:
        tuple: Симплекс з максимальним об'ємом та його об'єм.
    """
    max_volume = 0
    max_volume_simplex = None

    for simplex in itertools.combinations(points, 4):
        volume = calculate_tetrahedron_volume(np.array(simplex))
        if volume > max_volume:
            max_volume = volume
            max_volume_simplex = simplex

    return max_volume_simplex, max_volume
