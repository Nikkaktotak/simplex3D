# main.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
import buildModelLib as ml

# Генерація n випадкових точок в просторі
n = 0 #к-ть точок всередині кулі
m = 100 #к-ть точок на сфері
r = 1000
points, V = ml.generatePoints(n, m, r)

vertices, edges, faces = ml.createConvexHull(points)


