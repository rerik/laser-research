from lib import *


# 660 нм
Layers_660 = [
    Layer(),
    Layer(1.4, 0.06, 20, 0.82),
    Layer(1.55, 0.035, 36, 0.925),
    Layer(1.4, 0.001, 0.1, 0.999),
    Layer(1.4, 0.05, 60, 0.95),
    Layer(1.4, 0.02, 50, 0.8),
    Layer()
]

# 830 нм
Layers_830 = [
    Layer(),
    Layer(1.4, 0.05, 15, 0.86),
    Layer(1.55, 0.025, 28, 0.94),
    Layer(1.4, 0.001, 0.1, 0.999),
    Layer(1.4, 0.03, 60, 0.96),
    Layer(1.4, 0.01, 55, 0.85),
    Layer()
]

thicknesses = [0, 3, 10, 2, 4, 20]
biases = [0] * len(thicknesses)
for i in range(len(thicknesses)):
    biases[i] += biases[i - 1] - thicknesses[i]

Rectangles = [Rectangle(Vertex(-80, -50, t), Vertex(-80, 50, t), Vertex(80, 50, t)) for t in biases]

radius = 120
resolution = '40x25'

Boundaries = [Boundary(Boundary.create_spherical_boundary(R, radius, resolution)) for R in Rectangles]

research_area = Parallelepiped(Vertex(-80, -50, -100), Vector(160, 100, 110), "80x50x55")
