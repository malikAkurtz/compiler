from __future__ import annotations
from typing import Optional
import numpy as np
from core.constants import REDUCED_FLUX_QUANTUM

######################################## NODE CLASS ########################################
class Node:
    _id = 0
    def __init__(self, branches: Optional[list[Branch]] = None):
        self.id = Node._id
        Node._id  += 1
        
        # e.g [capacitor1, inductor1, JJ1, ...]
        self.branches = branches if branches is not None else []

    def _get_degree(self):
        # number of connected capacitive branches
        capacitive_degree = 0
        # number of connected inductive branches
        inductive_degree = 0
        
        for branch in self.branches:
            if isinstance(branch, CapacitiveElement):
                capacitive_degree += 1
            if isinstance(branch, InductiveElement):
                inductive_degree += 1
                
        return capacitive_degree, inductive_degree    
######################################## NODE CLASS ######################################## 

######################################## BRANCH CLASS ########################################
# A BRANCH IS AN ELEMENT, TWO NODES CAN BE CONNECTED BY MULTIPLE BRANCHES
class Branch:
    def __init__(self, nodes: Optional[tuple[Node]] = None):
        self.nodes = nodes
        
    def calculate_energy(self):
        pass
######################################## BRANCH CLASS ########################################

######################################## CAPACITIVE ELEMENT CLASS ########################################
class CapacitiveElement(Branch):
    def __init__(self, capacitance: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        self.C = capacitance
        
    def calculate_energy(self, branch_charge: float, branch_charge_offset: float):
        return ((branch_charge - branch_charge_offset)**2) / (2 * self.C)
######################################## CAPACITIVE ELEMENT CLASS ########################################
        
######################################## CAPACITOR  CLASS ########################################
class Capacitor(CapacitiveElement):
    def __init__(self, capacitance: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(capacitance, nodes)
######################################## CAPACITOR ELEMENT CLASS ########################################

######################################## INDUCTIVE ELEMENT CLASS ########################################
class InductiveElement(Branch):
    def __init__(self, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        
    def calculate_energy(self):
        pass
######################################## INDUCTIVE ELEMENT CLASS ########################################

######################################## INDUCTOR CLASS ########################################
class Inductor(InductiveElement):
    def __init__(self, inductance: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        self.L = inductance
        
    def calculate_energy(self, branch_flux: float, branch_flux_offset: float):
        return ((branch_flux - branch_flux_offset)**2) / (2 * self.L)
######################################## INDUCTOR CLASS ########################################

######################################## JOSEPHSON ELEMENT CLASS ########################################
class JosephsonElement(InductiveElement):
    def __init__(self, josephson_energy: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        self.EJ = josephson_energy
        
    def calculate_energy(self, branch_flux: float, branch_flux_offset: float):
        return -self.EJ * np.cos( (branch_flux - branch_flux_offset) / REDUCED_FLUX_QUANTUM)
######################################## JOSEPHSON ELEMENT CLASS ########################################

######################################## GRAPH CLASS ########################################
class Graph:
    def __init__(self, vertices: list[Node], edges: list[Branch]):
        self.vertices = vertices
        self.edges = edges
    
    @staticmethod
    def construct_graph(graph_rep: dict) -> Graph:
        """
        Build the full Graph object from the user-supplied dictionary.
        Also breaks symmetry by adding a tiny capacitor to ground for
        any node that lacks a capacitive branch.
        """
        gnd = Node(label="gnd", branches=None)
        label_to_node = {"gnd": gnd}
        branches = []

        # Create the node objects from the circuit dictionary
        for label in graph_rep["nodes"]: # gnd not included
            label_to_node[label] = Node(label=label, branches=None)

        # Create the capacitor objects from the circuit dictionary
        # and add it to the appropriate nodes
        for (node1_label, node2_label, capacitance) in graph_rep['capacitors']:
            node1 = label_to_node[node1_label]
            node2 = label_to_node[node2_label]
            
            capacitor = Capacitor(capacitance=capacitance, nodes=(node1, node2))
            
            branches.append(capacitor)
            node1.branches.append(capacitor)
            node2.branches.append(capacitor)
            
        # Create the linear inductor objects from the circuit dictionary
        # and add it to the appropriate nodes
        for (node1_label, node2_label, inductance) in graph_rep['inductors']:
            node1 = label_to_node[node1_label]
            node2 = label_to_node[node2_label]
            
            inductor = Inductor(inductance=inductance, nodes=(node1, node2))
            
            branches.append(inductor)
            node1.branches.append(inductor)
            node2.branches.append(inductor)

        # Create the Josephson element objects from the circuit dictionary
        # and add it to the appropriate nodes
        for (node1_label, node2_label, josephson_energy) in graph_rep['josephson_elements']:
            node1 = label_to_node[node1_label]
            node2 = label_to_node[node2_label]
            
            josephson_element = JosephsonElement(josephson_energy=josephson_energy, nodes=(node1, node2))
            
            branches.append(josephson_element)
            node1.branches.append(josephson_element)
            node2.branches.append(josephson_element)

        # Break symmetry: add tiny parasitic capacitor to ground for nodes with no
        # capacitive branch (prevents singular capacitance matrix)
        for node in label_to_node.values():
            if node.label == "gnd":
                continue
            has_capacitor = any(isinstance(b, CapacitiveElement) for b in node.branches)
            if not has_capacitor:
                capacitor = Capacitor(capacitance=1e-20, nodes=(node, gnd))
                
                branches.append(capacitor)
                node.branches.append(capacitor)
                gnd.branches.append(capacitor)

        return Graph(list(label_to_node.values()), branches)
######################################## GRAPH CLASS ########################################