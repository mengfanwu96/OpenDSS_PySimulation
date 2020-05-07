import numpy as np

# the Bus class uses the circuit instance from the DSS COM object
# and gets the names, bus voltages, distances from the energy meter,
# x and y coordinates of the 'from' bus and puts them into numpy arrays
class Bus:
    def __init__(self, circuit):
        """
        Inputs:
            circuit - DSS COM object
        Contains:
            name - string - bus name
            V - complex array (n x 3) - node bus voltage
            distance - array (n) - distance from the energymeter
            x - array (n) - from-bus x location
            y - array (n) - from-bus y location
        """
# n is set to the number of buses in the circuit
        n = circuit.NumBuses

# make the x,y, distance, and voltage numpy arrays of length n and set
# the vslues to all zeros
# note:  the voltage array is an array of complex values
#         x = np.zeros(n)
#         y = np.zeros(n)
#         distance = np.zeros(n)
        V = np.zeros((n,3), dtype=complex)
        name = np.array("                                ").repeat(n)

# populate the arrays by looking at the each bus in turn from 0 to n
# note:  by convention all arrays are zero-based in python
        for i in range(0,n):
            bus = circuit.Buses(i)
            name[i] = bus.Name
            # x[i] = bus.x
            # y[i] = bus.y
            # distance[i] = bus.Distance
            v = np.array(bus.Voltages)
            nodes = np.array(bus.Nodes)

# we're only interested in the first three nodes
# (also called terminals) on the bus
            if nodes.size > 3: nodes = nodes[0:3]
            cidx = 2 * np.array(range(0, min(v.size // 2, 3)))
            V[i, nodes-1] = v[cidx] + 1j * v[cidx + 1]
        self.name = name
        self.V = V
        # self.x = x
        # self.y = y
        # self.distance = distance


# Branch class contains the branch object details
class Branch:
    def __init__(self, circuit):
        """
        Inputs:
            circuit - DSS COM object
        Contains:
            name - string - branch name
            busname - string (n) - from-node bus name
            busnameto - string (n) - to-node bus name
            V - complex array (n x 3) - from-node bus voltage
            Vto - complex array (n x 3) - to-node bus voltage
            I - complex array (n x 3) - branch currents
            nphases - array (n) - number of phases
            distance - array (n) - distance from the energy meter
            x - array (n) - from-bus x location
            y - array (n) - from-bus y location
            xto - array (n) - to-bus x location
            yto - array (n) - to-bus y location
        """
        n = circuit.NumCktElements
        name = np.array("                      ").repeat(n)
        busname = np.array("                      ").repeat(n)
        busnameto = np.array("                      ").repeat(n)
        # x = np.zeros(n)
        # y = np.zeros(n)
        # xto = np.zeros(n)
        # yto = np.zeros(n)
        # distance = np.zeros(n)
        nphases = np.zeros(n)
        kvbase = np.zeros(n)
        I = np.zeros((n,3), dtype=complex)
        V = np.zeros((n,3), dtype=complex)
        Vto = np.zeros((n,3), dtype=complex)
        i = 0
        for j in range(0,n):
            el = circuit.CktElements(j)
            if not re.search("^Line", el.Name):
                continue  # only pick lines...
            name[i] = el.Name
            bus2 = circuit.Buses(re.sub(r"\..*","", el.BusNames[-1]))
            busnameto[i] = bus2.Name
            # xto[i] = bus2.x
            # yto[i] = bus2.y
            if bus2.x == 0 or bus2.y == 0: continue # skip lines without proper bus coordinates
            # distance[i] = bus2.Distance
            v = np.array(bus2.Voltages)
            nodes = np.array(bus2.Nodes)
            kvbase[i] = bus2.kVBase
            nphases[i] = nodes.size
            if nodes.size > 3: nodes = nodes[0:3]
            cidx = 2 * np.array(range(0, min(v.size // 2, 3)))

            bus1 = circuit.Buses(re.sub(r"\..*","", el.BusNames[0]))

            if bus1.x == 0 or bus1.y == 0:
                continue # skip lines without proper bus coordinates

            busname[i] = bus1.Name

            Vto[i, nodes-1] = v[cidx] + 1j * v[cidx + 1]
            # x[i] = bus1.x
            # y[i] = bus1.y
            v = np.array(bus1.Voltages)
            V[i, nodes-1] = v[cidx] + 1j * v[cidx + 1]
            current = np.array(el.Currents)
            I[i, nodes-1] = current[cidx] + 1j * current[cidx + 1]
            i = i + 1
        self.name = name[0:i]
        self.busname = busname[0:i]
        self.busnameto = busnameto[0:i]
        self.nphases = nphases[0:i]
        self.kvbase = kvbase[0:i]
        # self.x = x[0:i]
        # self.y = y[0:i]
        # self.xto = xto[0:i]
        # self.yto = yto[0:i]
        # self.distance = distance[0:i]

        self.V = V[0:i]
        self.Vto = Vto[0:i]
        self.I = I[0:i]