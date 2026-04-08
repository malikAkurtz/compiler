from Branch import Node, Capacitor
from DCSQUID import DCSQUID

class TransmonCircuit():
    def __init__(self, C_S: float, dcsquid: DCSQUID) -> None:
        self.gnd = dcsquid.gnd
        self.island = dcsquid.island
        self.branches = list(dcsquid.branches)
        
        # Just add a shunt capacitor
        capacitor = Capacitor(capacitance=C_S, nodes=[dcsquid.island, dcsquid.gnd])
        
        self.branches.append(capacitor)
        self.island.branches.append(capacitor)
        self.gnd.branches.append(capacitor)