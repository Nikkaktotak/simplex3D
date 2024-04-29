import numpy as np


def generatePoints(n, m, r):
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
    # Вершини тетраедра
    pyramid_points = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    pyramid_points = r * pyramid_points / np.linalg.norm(pyramid_points, axis=1)[:, np.newaxis]

    # Об'єднання всіх точок
    points = np.vstack((internal_points, sphere_points, pyramid_points))
    return points
