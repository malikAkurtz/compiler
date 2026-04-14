from Circuit import Circuit
from Branch import *

class LCResonatorCircuit(Circuit):
    def __init__(self, gnd: Node, capacitance: float, inductance: float):
        self.branches = []
        self.island   = Node(branches=[])
        
        capacitor = Capacitor(
            capacitance=capacitance,
            nodes=[gnd, self.island]
        )
        
        self.branches.append(capacitor)
        self.island.branches.append(capacitor)
        gnd.branches.append(capacitor)
        
        inductor = Inductor(
            inductance=inductance,
            nodes=[gnd, self.island]
        )
        
        self.branches.append(inductor)
        self.island.branches.append(inductor)
        gnd.branches.append(inductor)
        
        self.graph = Graph(vertices=[gnd, self.island], edges=self.branches)
        
        super().__init__(graph=self.graph)