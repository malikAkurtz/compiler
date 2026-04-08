
from Branch import *

class DCSQUID():
    _id = 0
    def __init__(self, gnd: Node, C_S: float, EJ_eff, C_JL: float, C_JR: float, C_C: float):
        self.island = Node(label=str(f"q_{DCSQUID._id}"), branches=[])
        DCSQUID._id += 1
        
        self.branches = []
        
        # Create capacitors
        for capacitance in [C_S, C_JL, C_JR, C_C]:
            if capacitance > 0:
                capacitor = Capacitor(capacitance=capacitance, nodes=[self.island, gnd])
                # Add capacitor to the DCSQUID object
                self.branches.append(capacitor)
                # Add capacitor to the nodes it connects
                self.island.branches.append(capacitor)
                gnd.branches.append(capacitor)
        
        # Create Josephson elements
        josephson_element = JosephsonElement(josephson_energy=EJ_eff, nodes=[self.island, gnd])
        
        # Add Josephson element to the DCSQUID object
        self.branches.append(josephson_element)
        # Add Josephson element to the branches it connects
        self.island.branches.append(josephson_element)
        gnd.branches.append(josephson_element)