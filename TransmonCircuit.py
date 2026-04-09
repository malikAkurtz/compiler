from Branch import Node, Capacitor
from DCSQUID import DCSQUID

class TransmonCircuit():
    def __init__(self, dcsquid: DCSQUID, shunt_capacitance: float, coupling_capacitance: float) -> None:
        self.gnd = dcsquid.gnd
        self.island = dcsquid.island
        self.branches = list(dcsquid.branches)
        
        # Add a shunt capacitor
        shunt_capacitor = Capacitor(capacitance=shunt_capacitance, nodes=[self.island, self.gnd])
        
        self.branches.append(shunt_capacitor)
        self.island.branches.append(shunt_capacitor)
        self.gnd.branches.append(shunt_capacitor)
        
        # Add a coupling capacitor
        coupling_capacitor = Capacitor(capacitance=coupling_capacitance, nodes=[self.island, self.gnd])
        
        self.branches.append(coupling_capacitor)
        self.island.branches.append(coupling_capacitor)
        self.gnd.branches.append(coupling_capacitor)
        
        