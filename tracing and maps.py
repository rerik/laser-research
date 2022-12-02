import matplotlib.pyplot as plt
from multiprocessing import Process
from os import cpu_count
from model import *


if __name__ == '__main__':

    Layers = Layers_660

    # рисование поверхностей, траекторий и карт распределения энергии
    for i in range(len(Boundaries)):
        Boundaries[i].set_between(Layers[i], Layers[i + 1])

    bias = 40  # Смещение детектора относительно источника
    detector = Detector(Vertex(bias, -40, 0), Vertex(bias, 40, 0), Vertex(bias + 80, 40, 0))

    detected_photons = Manager().list()

    processes = cpu_count() - 2
    print("Cores:", processes)
    photons_num = 200
    photons_per_process = int(photons_num / processes)

    procs = []
    for i in range(processes):
        p = Process(
            target=worker,
            args=(Layers, Boundaries, research_area, detector, detected_photons, photons_per_process))
        procs.append(p)
        p.start()

    [proc.join() for proc in procs]

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set(xlabel='X', ylabel='Y', zlabel='Z')
    for B in Boundaries:
        ax1.plot_trisurf(*B.get_line_3D())

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set(xlabel='X', ylabel='Y', zlabel='Z')
    for photon in detected_photons:
        line = photon.trajectory_line
        ax2.plot(*line)
        research_area.add_w(photon.trajectory[1:], photon.trajectory_w[1:])

    ax2.plot_trisurf(*detector.get_line_3D())

    print(detector.w.value)

    fig3 = plt.figure(3)
    plt.subplot(2, 2, 1)
    plt.imshow(research_area.pojection_x())

    plt.subplot(2, 2, 2)
    plt.imshow(research_area.pojection_y())

    plt.subplot(2, 2, 3)
    plt.imshow(research_area.pojection_z())

    # print(Vector.__doc__)

    plt.show()
