import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_bus_phase_step(name: str, attr: str):
    loc_step = name.split(attr)
    bus = loc_step[0].split('.')[0]
    phase = int(loc_step[0].split('.')[1])
    step = int(loc_step[1])
    return bus, phase, step


def get_bus_phase(bus: str):
    bus_loc = bus.split('.')
    rd = {}
    if len(bus_loc) == 1:
        rd[bus_loc[0]] = 'all'
    else:
        rd[bus_loc[0]] = []
        for i in range(1, len(bus_loc)):
            rd[bus_loc[0]].append(int(bus_loc[i]))
    return rd


with open('scores.csv') as r:
    data = pd.read_csv(r)

    attr_names = [ 'bus', 'phase', 'attr']
    value_names = ['value%s' % i for i in range(10)]
    columns = attr_names + value_names


    shape = data.shape
    points_num = shape[0]

    crit = ['fr', 'mutual_info']
    buses = ['671.1.2.3', '684.1.3', '652.1']
    attr = ['v', 'c', 'theta', 'phi']

    for idx, cri in enumerate(crit):
        for bus_loc in buses:
            res = pd.DataFrame(columns=columns)
            bus_info = get_bus_phase(bus_loc)
            for bus, phases in bus_info.items():
                for phase in phases:
                    for a in attr:
                        vec = np.zeros(10)
                        for step in range(10):
                            name = bus + '.' + str(phase) + a + str(step)
                            vec[step] = data[name][idx]
                        a_vec = np.array([bus, phase, a])
                        b = np.hstack((a_vec, vec)).reshape(1, -1)
                        res = res.append(pd.DataFrame(data=b, columns=columns))

            fig, ax = plt.subplots()
            for index, row in res.iterrows():
                vector = np.array(row[value_names], dtype=float)
                types = str(row[attr_names].values)
                print(types + str(np.average(vector[-5:-1])))
                ax.plot(np.arange(10), vector, label=str(types))
            ax.legend()
            plt.show()


    for idx, cri in enumerate(crit):
        for a in attr:
            res = pd.DataFrame(columns=columns)
            for bus_loc in buses:
                bus_info = get_bus_phase(bus_loc)
                for bus, phases in bus_info.items():
                    for phase in phases:
                        vec = np.zeros(10)
                        for step in range(10):
                            name = bus + '.' + str(phase) + a + str(step)
                            vec[step] = data[name][idx]
                        a_vec = np.array([bus, phase, a])
                        b = np.hstack((a_vec, vec)).reshape(1, -1)
                        res = res.append(pd.DataFrame(data=b, columns=columns))

            fig, ax = plt.subplots()
            for index, row in res.iterrows():
                vector = np.array(row[value_names], dtype=float)
                types = str(row[attr_names].values)
                print(types + str(np.average(vector[-5:-1])))
                ax.plot(np.arange(10), vector, label=str(types))
            ax.legend()
            plt.show()
