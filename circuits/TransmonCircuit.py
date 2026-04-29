from core.Branch import Node, Capacitor, Graph
from circuits.DCSQUIDCircuit import DCSQUIDCircuit

class TransmonCircuit():
    def __init__(self, gnd: Node, dcsquid: DCSQUIDCircuit, shunt_capacitance: float, coupling_capacitance: float) -> None:
        self.dcquid = dcsquid
        
        self.island = dcsquid.island
        self.branches = list(dcsquid.branches)
        self.CC = coupling_capacitance
        
        # Add a shunt capacitor
        shunt_capacitor = Capacitor(capacitance=shunt_capacitance, nodes=[dcsquid.island, gnd])
        
        self.branches.append(shunt_capacitor)
        self.island.branches.append(shunt_capacitor)
        gnd.branches.append(shunt_capacitor)
        
        # Add a coupling capacitor
        coupling_capacitor = Capacitor(capacitance=coupling_capacitance, nodes=[dcsquid.island, gnd])
        
        self.branches.append(coupling_capacitor)
        self.island.branches.append(coupling_capacitor)
        gnd.branches.append(coupling_capacitor)
        
        self.graph = Graph(vertices=[gnd, dcsquid.island], edges=self.branches)
        
        