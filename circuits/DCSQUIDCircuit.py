
from core.Branch import *
from core.Circuit import Circuit
class DCSQUIDCircuit(Circuit):
    def __init__(self, gnd: Node, external_flux: float, left_josephson_energy: float, right_josephson_energy: float, left_josephson_capacitance: float, right_josephson_capacitance: float) -> None:
        self.EJL = left_josephson_energy
        self.EJR = right_josephson_energy
        
        # ---- Create the Island, Stores Net Guage-Invariant Phase Difference Across the Josephson Junction ----
        self.island = Node(branches=[])
        
        # ---- Create branches belonging to the DCSQUID ----
        self.branches = []
        
        # ---- Create capacitors ----
        for capacitance in [left_josephson_capacitance, right_josephson_capacitance]:
            if capacitance > 0:
                capacitor = Capacitor(capacitance=capacitance, nodes=[self.island, gnd])
                
                # Add capacitor to the DCSQUID
                self.branches.append(capacitor)

                # Add capacitor to the nodes it connects
                self.island.branches.append(capacitor)
                gnd.branches.append(capacitor)
                
        # ---- Calculate Effective Josphson Energy as a Function of PHI_ext ----
        EJ = DCSQUIDCircuit.calculate_effective_EJ(external_flux, self.EJL, self.EJR)
        
        # ---- Create a Josephson element that plays the role of both Josephson elements with EJ_eff ----
        josephson_element = JosephsonElement(josephson_energy=EJ, nodes=[self.island, gnd])
        
        # Add Josephson element to the DCSQUID
        self.branches.append(josephson_element)
        
        # Add Josephson element to the branches it connects
        self.island.branches.append(josephson_element)
        gnd.branches.append(josephson_element)
        
        self.graph = Graph(vertices=[gnd, self.island], edges=self.branches)
        
        super().__init__(graph=self.graph)
        
    @staticmethod
    def calculate_effective_EJ(PHI_ext: float, EJL: float, EJR: float):
        EJ_sum = EJL + EJR
        d   = (EJR - EJL) / (EJ_sum)
        
        return EJ_sum * np.cos(PHI_ext / (2 * REDUCED_FLUX_QUANTUM)) * np.sqrt(1 + (d**2) * (np.tan(PHI_ext / (2 * REDUCED_FLUX_QUANTUM))**2))
            
        