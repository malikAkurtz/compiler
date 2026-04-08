
from Branch import *

class DCSQUID():
    _id = 0
    def __init__(self, gnd: Node, PHI_ext: float, JL: float, JR: float, C_JL: float, C_JR: float, C_C: float) -> None:
        # ---- Store Alias to Ground Node ----
        self.gnd = gnd
        
        # ---- Calculate Effective Josphson Energy as a Function of PHI_ext ----
        EJ_eff = DCSQUID.calculate_effective_EJ(PHI_ext, JL, JR)
        
        # ---- Create the Island, Stores Net Guage-Invariant Phase Difference Across the Josephson Junction ----
        self.island = Node(label=str(f"q_{DCSQUID._id}"), branches=[])
        DCSQUID._id += 1
        self.branches = []
        
        # ---- Create capacitors ----
        for capacitance in [C_JL, C_JR, C_C]:
            if capacitance > 0:
                capacitor = Capacitor(capacitance=capacitance, nodes=[self.island, self.gnd])
                # Add capacitor to the DCSQUID object
                self.branches.append(capacitor)
                # Add capacitor to the nodes it connects
                self.island.branches.append(capacitor)
                self.gnd.branches.append(capacitor)
        
        # ---- Create a Josephson element that plays the role of both Josephson elements with EJ_eff ----
        josephson_element = JosephsonElement(josephson_energy=EJ_eff, nodes=[self.island, self.gnd])
        
        # Add Josephson element to the DCSQUID object
        self.branches.append(josephson_element)
        # Add Josephson element to the branches it connects
        self.island.branches.append(josephson_element)
        self.gnd.branches.append(josephson_element)
        
    
    @staticmethod
    def calculate_effective_EJ(external_flux: float, JL: float, JR: float):
        EJL = REDUCED_FLUX_QUANTUM * JL
        EJR = REDUCED_FLUX_QUANTUM * JR
        
        d   = (JR - JL) / (JR + JL)
        
        return (EJL + EJR) * np.cos(external_flux / (2 * REDUCED_FLUX_QUANTUM)) * np.sqrt(1 + (d**2) * (np.tan(external_flux / (2 * REDUCED_FLUX_QUANTUM))**2))
            
        