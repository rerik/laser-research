from __future__ import annotations
from typing import List, Tuple
from math import e, pi, sin, cos, sqrt, asin, acos, tan
from random import random, uniform
from numpy import dot, cross, linspace
from numpy.linalg import norm
from copy import copy
from multiprocessing import Manager, current_process


INFINITY = 100


def get_random_num_by_function(max_value: float = 1, beg: float = 0, end: float = 1, function=lambda x: x, *args) -> float:
    x = uniform(beg, end)
    while function(*args, x) < max_value * random():
        x = random()
    return x


class Vector:

    """
    Класс описывает вектор, задаваемый 3-мя компонентами: x, y, z
    Вызов экземпляра A() возвращает компоненты списком [x, y, z]
    Для векторов определён ряд операторов:
    A+B - сумма векторов
    A-B - разность векторов
    A -= B - уменшение вектора
    -B - обратный вектор
    A*b - умножение вектора на число
    A*B - скалярное произведение векторов
    A/b и A /= b - деление вектора на число
    A**B - векторное произведение векторов
    A|B - нормаль к плоскости, заданной векторами
    A^B - угол между векторами
    !!! Класс не универсальный, определены лишь необходимые операции !!!
    Определена Евклидова величина (модуль) вектора - A.length
    В условии нулевой вектор возвращает False, любой иной - True
    Функция print(A) выведет компоненты вектора в формате x|y|z
    """

    x:  float
    y:  float
    z:  float

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __call__(self) -> list:
        return [self.x, self.y, self.z]

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __isub__(self, other: Vector) -> Vector:
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __neg__(self) -> Vector:
        return Vector(-self.x, -self.y, -self.z)

    def __mul__(self, other: Vector | int | float) -> float | Vector:
        if type(other) == Vector:
            return dot(self(), other())
        else:
            return Vector(self.x*other, self.y*other, self.z*other)

    def __truediv__(self, other: int | float) -> Vector:
        return Vector(self.x / other, self.y / other, self.z / other)

    def __idiv__(self, other: int | float) -> Vector:
        self.x /= other
        self.y /= other
        self.z /= other
        return self

    def __pow__(self, other: Vector) -> Vector:
        x, y, z = cross(self(), other())
        return Vector(x, y, z)

    @property
    def length(self) -> float:
        return sqrt(self*self)

    def __or__(self, other: Vector) -> Vector:
        vector = self**other
        return vector/vector.length

    def __bool__(self) -> bool:
        return bool(self.x or self.y or self.z)

    def __repr__(self) -> str:
        return f"{self.x}|{self.y}|{self.z}"

    def __xor__(self, other: Vector) -> float:
        return acos(self*other/self.length/other.length)


class Vertex:

    """
    Класс описывает точку, задаваемую 3-мя координатами: x, y, z
    Вызов экземпляра A() возвращает координаты списком [x, y, z]
    Для точек определён ряд операторов:
    A-B - вектор от точки B к точке A
    P-V и P -= V - перемещение точки вычитанием вектора
    P+V и P += V - перемещение точки прибавлением вектора
    -B - радиус-вектор точки
    !!! Класс не универсальный, определены лишь необходимые операции !!!
    Функция print(A) выведет координаты точки в формате x:y:z
    """

    x:  float
    y:  float
    z:  float

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __call__(self) -> list:
        return [self.x, self.y, self.z]

    def __sub__(self, other: Vertex | Vector) -> Vector | Vertex:
        if type(other) == Vertex:
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

    def __isub__(self, other: Vector) -> Vertex:
        return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: Vector) -> Vertex:
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    def __iadd__(self, other: Vector) -> Vertex:
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __neg__(self) -> Vector:
        return Vector(self.x, self.y, self.z)

    def __repr__(self) -> str:
        return f"{round(self.x, 5)}:{round(self.y, 5)}:{round(self.z, 5)}"


class Layer:

    """
    Класс моделирует слой исследуемой среды.
    Он описывает лишь физические свойства среды, но не геометрию.
    Геометрию описывает класс границ Boundary ниже по коду.
    """

    n: float  # коэффициент преломления среды
    ua: float  # коэффициент поглощения среды
    us: float  # коэффициент рассеяния среды
    g: float  # коэффициент анизотропии среды

    def __init__(self, n: float = 1, ua: float = 0, us: float = 0, g: float = 1) -> None:
        self.n = n
        self.ua = ua
        self.us = us
        self.g = g

    @property
    def ut(self) -> float: return self.ua + self.us


class Parallelepiped:

    """
    Класс моделирует область пространства в форме параллелипипеда,
    а также распределение энергии в нём в виде трёхмерной сетки
    с заданным разрешением с распределением весов.

    Область задаётся начальной точкой и вектором. Разрешение
    задаётся строкой в формате "XxYxZ".

    Функция is_inside позволяет определить, находится ли
    точка внутри заданной области пространства.

    Функция add_w аппроксимирует для каждой точки траектории фотона
    ячейку трёхмерной сетки и добавляет к ней количество
    рассеянной фотоном энергии в текущей точке.

    Функции pojection_x, pojection_y и pojection_z позволяют
    получить двумерные проекции сетки для дальнейшего отображения
    и визуализации.
    """

    origin: Vertex
    final: Vertex
    size: Vector

    resolution_x: int
    resolution_y: int
    resolution_z: int

    def __init__(self, origin: Vertex, size: Vector, resolution: str = "10x10x10") -> None:
        self.origin = origin
        self.size = size
        self.final = origin + size
        self.resolution_x, self.resolution_y, self.resolution_z = map(int, resolution.split('x'))
        self.grid = [[[0 for i in range(self.resolution_z)]
                      for j in range(self.resolution_y)]
                     for k in range(self.resolution_x)]

    @staticmethod
    def is_between(value: float, value_1: float, value_2: float) -> bool:
        min_value, max_value = (value_1, value_2) if value_1 < value_2 else (value_2, value_1)
        return min_value <= value <= max_value

    def is_inside(self, v: Vertex) -> bool:
        condition_x = self.is_between(v.x, self.origin.x, self.final.x)
        condition_y = self.is_between(v.y, self.origin.y, self.final.y)
        condition_z = self.is_between(v.z, self.origin.z, self.final.z)
        return condition_x and condition_y and condition_z

    def add_w(self, trajectory: List[Vertex], trajectory_w: List[float]) -> None:
        bias = -self.origin
        for point, w in zip(trajectory, trajectory_w):
            point -= bias
            x = int(round(point.x/self.size.x*self.resolution_x))-1
            y = int(round(point.y/self.size.y*self.resolution_y))-1
            z = int(round(point.z/self.size.z*self.resolution_z))-1
            self.grid[x][y][z] += w

    def pojection_x(self) -> List[List[float]]:
        return [[sqrt(sum([self.grid[i][j][k] for i in range(self.resolution_x)]))
                 for j in range(self.resolution_y)]
                for k in range(self.resolution_z)]

    def pojection_y(self) -> List[List[float]]:
        return [[sqrt(sum([self.grid[i][j][k] for j in range(self.resolution_y)]))
                 for i in range(self.resolution_x)]
                for k in range(self.resolution_z)]

    def pojection_z(self) -> List[List[float]]:
        return [[sqrt(sum([self.grid[i][j][k] for k in range(self.resolution_z)]))
                 for i in range(self.resolution_x)]
                for j in range(self.resolution_y)]


class TrianglesSequence:

    vertices: List[Vertex]  # совокупность вершин, описывающих границу слоя методом триангуляции
    length: int  # количество точек в последовательности
    triangles_num: int  # количество треугольников, задающих поверхность

    def __init__(self, vertices: List[Vertex]) -> None:
        self.vertices = vertices
        self.length = len(vertices)
        self.triangles_num = self.length - 2

    # Алгоритм Моллера — Трумбора
    def triangle_intersection(self, triangle_number: int, origin: Vertex, direction: Vector) -> float:

        v0 = self.vertices[triangle_number]
        v1 = self.vertices[triangle_number + 1]
        v2 = self.vertices[triangle_number + 2]

        e1 = v1 - v0
        e2 = v2 - v0

        pvec = direction ** e2
        det = e1 * pvec

        if det == 0:
            return 0

        inv_det = 1 / det
        tvec = origin - v0
        u = tvec * pvec * inv_det
        if u < 0 or u > 1:
            return 0

        qvec = tvec ** e1
        v = direction * qvec * inv_det
        if v < 0 or u + v > 1:
            return 0

        return e2 * qvec * inv_det

    def triangle_normal(self, triangle_number: int) -> Vector:
        v0 = self.vertices[triangle_number]
        v1 = self.vertices[triangle_number + 1]
        v2 = self.vertices[triangle_number + 2]

        e1 = v1 - v0
        e2 = v2 - v0

        return e1 | e2

    def intersection(self, origin: Vertex, direction: Vector) -> Tuple[float, Vector]:
        min_distance = 0
        triangle_number = 0
        for i in range(self.triangles_num):
            distance = self.triangle_intersection(i, origin, direction)
            if distance:
                if min_distance:
                    if distance < min_distance:
                        min_distance = distance
                        triangle_number = i
                else:
                    min_distance = distance
                    triangle_number = i
        normal = self.triangle_normal(triangle_number) if min_distance else Vector()
        return min_distance, normal

    def get_line_3D(self) -> Tuple[List[float], List[float], List[float]]:
        x = [0.]*self.length
        y = [0.]*self.length
        z = [0.]*self.length
        for i in range(self.length):
            x[i] = self.vertices[i].x
            y[i] = self.vertices[i].y
            z[i] = self.vertices[i].z
        return x, y, z


class Rectangle:

    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex) -> None:
        self.origin = v2
        self.x_vector = v3 - v2
        self.y_vector = v1 - v2
        self.center = self.origin + self.x_vector/2 + self.y_vector/2
        self.width = self.x_vector.length
        self.height = self.y_vector.length
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v3 + (v1 - v2)

    def get_vertex_by_coords(self, x: float, y: float) -> Vertex:
        return self.origin + self.x_vector*x + self.y_vector*y

    def triangulation(self, resolution: str = '10x10') -> List[TrianglesSequence]:
        width, height = map(int, resolution.split('x'))
        x_lines = linspace(0, 1, num=height+1)
        y_lines = linspace(0, 1, num=width+1)

        result = []

        for line_num in range(height):
            first = [self.get_vertex_by_coords(x_lines[line_num+1], y_lines[0])]
            middle = [self.get_vertex_by_coords(x_lines[line_num+i % 2], y_lines[i]) for i in range(width+1)]
            last = [self.get_vertex_by_coords(x_lines[line_num+(width+1) % 2], y_lines[width])]
            result.append(TrianglesSequence(first + middle + last))

        return result

    def get_line_3D(self, resolution: str = '10x10', triangles_sequences: List[TrianglesSequence] = []) -> Tuple[List[float], List[float], List[float]]:
        if not triangles_sequences:
            triangles_sequences = self.triangulation(resolution)
        X = []
        Y = []
        Z = []
        for sequence in triangles_sequences:
            x, y, z = sequence.get_line_3D()
            X += x
            Y += y
            Z += z
        X = [self.v1.x] + [self.v4.x] + [self.v3.x] + [self.v2.x] + X
        Y = [self.v1.y] + [self.v4.y] + [self.v3.y] + [self.v2.y] + Y
        Z = [self.v1.z] + [self.v4.z] + [self.v3.z] + [self.v2.z] + Z
        return X, Y, Z


class Boundary:

    layers: List[Layer]  # слои, между которыми находится поверхность
    sequences: List[TrianglesSequence]

    def __init__(self, sequences: List[TrianglesSequence] | TrianglesSequence) -> None:
        self.sequences = sequences

    def set_between(self, layer_1, layer_2) -> None:
        self.layers = [layer_1, layer_2]

    def transaction(self, layer) -> Layer | None:
        if layer in self.layers:
            if layer is self.layers[0]:
                return self.layers[1]
            else:
                return self.layers[0]
        else:
            return None

    def intersection(self, origin: Vertex, direction: Vector) -> Tuple[float, Vector]:
        min_distance = 0
        min_normal = Vector()
        for sequence in self.sequences:
            distance, normal = sequence.intersection(origin, direction)
            if distance:
                if min_distance:
                    if distance < min_distance:
                        min_distance = distance
                        min_normal = normal
                else:
                    min_distance = distance
                    min_normal = normal
        return min_distance, min_normal

    @staticmethod
    def create_spherical_boundary(rectangle: Rectangle, radius: float, resolution: str = '10x10') -> List[TrianglesSequence]:
        if radius < rectangle.width/2 or radius < rectangle.height/2:
            print("Невозможно построить поверхность!")
        else:
            normal = rectangle.x_vector | rectangle.y_vector
            boundary = rectangle.triangulation(resolution)
            for sequence in boundary:
                for i in range(sequence.length):
                    distance = (sequence.vertices[i] - rectangle.center).length
                    sequence.vertices[i] += normal*(radius*(1-sqrt(1-(distance/radius)**2)))
            return boundary

    def get_line_3D(self):
        X = []
        Y = []
        Z = []
        for sequence in self.sequences:
            x, y, z = sequence.get_line_3D()
            X += x
            Y += y
            Z += z
        return X, Y, Z


class Photon:

    w: float
    position: Vertex
    direction: Vector
    trajectory: List[Vertex]
    trajectory_w: List[float]

    def __init__(self, origin: Vertex = Vertex(0, 0, 0), direction: Vector = Vector(0, 0, 0)) -> None:
        self.w = 1
        self.position = copy(origin)
        self.trajectory = [copy(origin)]
        self.trajectory_w = []
        direction_norm = norm(direction)
        self.direction = direction/direction_norm if direction_norm else copy(direction)

    @property
    def trajectory_line(self) -> Tuple[List[float], List[float], List[float]]:
        length = len(self.trajectory)
        X = [0.]*length
        Y = [0.]*length
        Z = [0.]*length
        for i, point in enumerate(self.trajectory):
            X[i] = point.x
            Y[i] = point.y
            Z[i] = point.z
        return X, Y, Z

    @staticmethod
    def Rsp(n1: float, n2: float, n3: float = None) -> float:
        r1 = (n1 - n2) ** 2 / (n1 + n2) ** 2
        if n3:
            r2 = (n2-n3)**2/(n2+n3)**2
            return r1 + (1-r1)**2*r2/(1-r1*r2)
        else:
            return r1

    def Wsp(self, n1: float, n2: float, n3: float = None) -> float:
        return 1 - self.Rsp(n1, n2, n3)

    @staticmethod
    def fs(ua: float, us: float, x: float = 1) -> float:
        ut = ua + us
        return ut*e**(-ut*x)

    @classmethod
    def get_random_num_by_fs(cls, layer: Layer) -> float:
        ua = layer.ua
        us = layer.us
        return get_random_num_by_function(ua+us, 0, INFINITY, cls.fs, ua, us)

    def move(self, scale: float = 1) -> None:
        self.position += self.direction*scale
        self.trajectory.append(copy(self.position))

    def w_update(self, ua: float, us: float) -> float:
        ut = ua + us
        if ut > 0:
            dw = self.w*ua/ut
            self.trajectory_w.append(dw)
            self.w -= dw
            return dw
        else:
            self.trajectory_w.append(0)
            return 0

    @staticmethod
    def fcostheta(g: float, x: float) -> float:
        return -g**2/(2*(1+g**2+2*g*x)**0.5)

    @staticmethod
    def get_random_num_by_fcostheta(g: float) -> float:
        xi = random()
        while xi == 1:
            xi = random()
        gg = g**2
        if g:
            return (1+gg-(1-gg)/(1-g+2*g*xi))/2/g
        else:
            return 2*xi-1

    def scattering(self, g: float) -> None:

        psi = random()*2*pi

        ct = self.get_random_num_by_fcostheta(g)
        st = sqrt(1-ct**2)
        sp = sin(psi)
        cp = cos(psi)

        v = sqrt(1-self.direction.z**2)

        vx = self.direction.x
        vy = self.direction.y
        vz = self.direction.z

        if abs(vz) > 0.9:
            self.direction.x = st*cp
            self.direction.y = st*sp
            if vz < 0:
                self.direction.z = ct
            else:
                self.direction.z = -ct
        else:
            self.direction.x = st/v * (vx*vz*cp - vy*sp) + vx*ct
            self.direction.y = st/v * (vy*vz*cp - vx*sp) + vy*ct
            self.direction.z = -st*cp*v + vz*ct

        self.direction /= self.direction.length

    @staticmethod
    def at(ai: float, ni: float, nt: float) -> float:
        var = ni*sin(ai)/nt
        if abs(var) > 1:
            return 2
        else:
            return asin(ni*sin(ai)/nt)

    @staticmethod
    def R(ai: float, at: float) -> float:
        sm = sin(ai - at)**2
        sp = sin(ai + at)**2
        tm = tan(ai - at)**2
        tp = tan(ai + at)**2
        return (sm/sp + tm/tp)/2

    @classmethod
    def is_reflected(cls, ai: float, at: float) -> bool:
        if at == 2:
            return True
        else:
            xi = random()
            while xi == 1:
                xi = random()
            return xi <= cls.R(ai, at)

    def reflect(self, ai: float, at: float, vn: Vector) -> bool:
        if self.is_reflected(ai, at):
            self.direction -= vn*(self.direction*vn*2)
            self.direction /= self.direction.length
            return True
        return False

    def transact(self, ai: float, at: float, ni: float, nt: float, nv: float, vn: Vector) -> bool:
        if not self.reflect(ai, at, vn):
            var1 = self.direction*ni*vn
            var2 = (nt**2-ni**2)/var1**2+1
            var2 = 0 if var2 < 0 else var2
            var3 = sqrt(var2)-1
            c = var1*var3
            self.direction = self.direction*nv + vn*c
            self.direction /= self.direction.length
            return True
        else:
            return False


class Detector(Boundary):

    rectangle: Rectangle
    # w: float

    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex) -> None:
        self.rectangle = Rectangle(v1, v2, v3)
        super().__init__(self.rectangle.triangulation('1x1'))
        self.w = Manager().Value('d', 0.)

    def detected(self, photon: Photon) -> bool:
        origin = photon.position
        direction = photon.direction
        res = bool(self.intersection(origin, direction)[0])
        if res:
            self.w.value += photon.w
        return res


def worker(
        Layers: List[Layer],
        Boundaries: List[Boundary],
        research_area: Parallelepiped,
        detector: Detector,
        detected_photons: List[Photon],
        num: int
) -> None:
    name_proc = current_process().name
    if len(name_proc) == 9:
        name_proc = name_proc+' '
    source = Vertex(0, 0, 1)
    direction = Vector(0, 0, -1)
    photons = [Photon(source, direction) for _ in range(num)]
    last_Boundary = None
    for photon in photons:
        boundary = None
        current_Layer = Layers[0]
        for i in range(2000):
            if photon.w < 0.001:
                print(i, photon.w)
                break
            if not research_area.is_inside(photon.position):
                print(name_proc, i, "out of area")
                break
            if current_Layer == Layers[0] and detector.detected(photon):
                photon.trajectory.append(
                    photon.position + photon.direction * detector.intersection(photon.position, photon.direction)[0])
                detected_photons.append(photon)
                print(name_proc, i, "detected")
                break
            elif current_Layer == Layers[-1]:
                print(name_proc, i, "out of area")
                break
            if i == 1999:
                print(name_proc, i, photon.w)
            min_distance = 0
            min_normal = Vector()
            for B in Boundaries:
                if current_Layer in B.layers and B != last_Boundary:
                    distance, normal = B.intersection(photon.position, photon.direction)
                    if distance:
                        if min_distance:
                            if distance < min_distance:
                                min_distance = distance
                                min_normal = normal
                                boundary = B
                        else:
                            min_distance = distance
                            min_normal = normal
                            boundary = B
            if min_distance:
                last_Boundary = boundary
                if current_Layer.n == 1:
                    photon.move(min_distance)
                    next_Layer = boundary.transaction(current_Layer)
                    photon.w_update(current_Layer.ua, current_Layer.us)
                    ai = min(photon.direction ^ (-min_normal), photon.direction ^ min_normal, key=lambda x: abs(x))
                    ni = current_Layer.n
                    nt = next_Layer.n
                    at = photon.at(ai, ni, nt)
                    nv = 1
                    vn = min_normal
                    if photon.transact(ai, at, ni, nt, nv, vn):
                        current_Layer = next_Layer
                else:
                    fs = photon.get_random_num_by_fs(current_Layer)
                    # print(f"Свободный пробег: {fs}")
                    if fs < min_distance:
                        photon.move(fs)
                        photon.w_update(current_Layer.ua, current_Layer.us)
                        photon.scattering(current_Layer.g)
                        last_Boundary = None
                    else:
                        photon.move(min_distance)
                        next_Layer = boundary.transaction(current_Layer)
                        photon.w_update(current_Layer.ua, current_Layer.us)
                        # print(f"next_Layer n: {next_Layer.n}")
                        ai = min(photon.direction ^ (-min_normal), photon.direction ^ min_normal, key=lambda x: abs(x))
                        ni = current_Layer.n
                        nt = next_Layer.n
                        at = photon.at(ai, ni, nt)
                        nv = 1
                        vn = min_normal
                        if photon.transact(ai, at, ni, nt, nv, vn):
                            current_Layer = next_Layer
            else:
                fs = photon.get_random_num_by_fs(current_Layer)
                photon.move(fs)
                photon.w_update(current_Layer.ua, current_Layer.us)
                photon.scattering(current_Layer.g)
                last_Boundary = None

