import matplotlib.pyplot as plt
from multiprocessing import Process
from os import cpu_count
from model import *


POSITIONS = 50


if __name__ == '__main__':

    # графики  энергии, улавлиемовой детекторами
    for i in range(len(Boundaries)):
        Boundaries[i].set_between(Layers_660[i], Layers_660[i+1])

    processes = cpu_count() - 2
    print("Cores:", processes)
    photons_num = 200
    photons_per_process = int(photons_num / processes)

    w_660 = []
    w_830 = []

    for i in range(POSITIONS):

        detected_photons = Manager().list()

        bias = i+1  # Смещение детектора относительно источника
        detector = Detector(Vertex(bias, -40, 0), Vertex(bias, 40, 0), Vertex(bias+80, 40, 0))

        procs = []
        for _ in range(processes):
            p = Process(
                target=worker,
                args=(Layers_660, Boundaries, research_area, detector, detected_photons, photons_per_process))
            procs.append(p)
            p.start()

        [proc.join() for proc in procs]

        w_660.append(detector.w.value)

    for i in range(len(Boundaries)):
        Boundaries[i].set_between(Layers_830[i], Layers_830[i+1])

    for i in range(POSITIONS):

        detected_photons = Manager().list()

        bias = i+1  # Смещение детектора относительно источника
        detector = Detector(Vertex(bias, -40, 0), Vertex(bias, 40, 0), Vertex(bias+80, 40, 0))

        procs = []
        for _ in range(processes):
            p = Process(
                target=worker,
                args=(Layers_830, Boundaries, research_area, detector, detected_photons, photons_per_process))
            procs.append(p)
            p.start()

        [proc.join() for proc in procs]

        w_830.append(detector.w.value)

    fig, ax = plt.subplots()
    ax.plot(w_660)
    ax.plot(w_830)
    plt.show()
