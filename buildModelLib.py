import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull

import numpy as np
from scipy.spatial import ConvexHull as SciPyConvexHull

class MyConvexHull:
    def __init__(self, points):
        self.points = points
        self.vertices = []
        self.edges = []
        self.faces = []
        self._compute_hull()

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

        # Recompute the convex hull for the neighborhood
        sub_hull = SciPyConvexHull(points[neighbors])

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

        # Map local indices back to global indices
        global_faces = []
        for face in new_faces:
            global_faces.append([neighbors[i] for i in face])

        global_edges = []
        for edge in new_edges:
            global_edges.append([neighbors[i] for i in edge])

        # Update the attributes directly
        self.vertices = points.tolist()
        self.edges = global_edges
        self.faces = global_faces


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
    convex_hull = MyConvexHull(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    poly3d = [np.array(convex_hull.vertices)[face] for face in convex_hull.faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.25))

    ax.scatter(np.array(convex_hull.vertices)[:, 0], np.array(convex_hull.vertices)[:, 1], np.array(convex_hull.vertices)[:, 2], color='b')

    max_range = np.array([np.array(convex_hull.vertices)[:, 0].max() - np.array(convex_hull.vertices)[:, 0].min(),
                          np.array(convex_hull.vertices)[:, 1].max() - np.array(convex_hull.vertices)[:, 1].min(),
                          np.array(convex_hull.vertices)[:, 2].max() - np.array(convex_hull.vertices)[:, 2].min()]).max() / 2.0

    mid_x = (np.array(convex_hull.vertices)[:, 0].max() + np.array(convex_hull.vertices)[:, 0].min()) * 0.5
    mid_y = (np.array(convex_hull.vertices)[:, 1].max() + np.array(convex_hull.vertices)[:, 1].min()) * 0.5
    mid_z = (np.array(convex_hull.vertices)[:, 2].max() + np.array(convex_hull.vertices)[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

    return convex_hull

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


