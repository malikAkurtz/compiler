###############################################################################
#
#   Circuit.py
#
#   Builds a circuit graph from a user-defined topology and constructs the
#   capacitance and inductance matrices.
#
###############################################################################

import numpy as np
from Branch import *

###############################################################################
#
#   Circuit Class
#
###############################################################################

class Circuit:
    """
    Represents a superconducting circuit as a graph.
    Responsibilities: graph construction, capacitance/inductance matrices,
    """
    # =====================================================================
    #   Constructor
    # =====================================================================

    def __init__(self, graph: Graph):
        self.circuit_graph                                    = graph
        self.labels                                           = [v.label for v in graph.vertices]
        self.capacitive_sub_graph, self.inductive_sub_graph   = self._build_sub_graphs()
        self.josephson_elements                               = [e for e in self.inductive_sub_graph.edges if isinstance(e, JosephsonElement)]
        self.active_nodes, self.passive_nodes                 = self._partition_nodes()
        self.N                                                = len(self.active_nodes)
        self.P                                                = len(self.active_nodes) + len(self.passive_nodes) + 1 # + 1 for ground
        self.capacitance_matrix, self.inv_inductance_matrix   = self._build_matrices()
        self.inv_capacitance_matrix                           = np.linalg.inv(self.capacitance_matrix)

    # =====================================================================
    #   Sub-Graph Construction
    # =====================================================================

    def _build_sub_graphs(self):
        """Split master graph into capacitive and inductive sub-graphs."""
        nodes    = self.circuit_graph.vertices
        branches = self.circuit_graph.edges
        # Capacitors
        capacitive_branches = [b for b in branches if isinstance(b, CapacitiveElement)]
        # Inductors and Josephson elements
        inductive_branches = [b for b in branches if isinstance(b, InductiveElement)]
        
        return Graph(nodes, capacitive_branches), Graph(nodes, inductive_branches)

    # =====================================================================
    #   Node Partitioning
    # =====================================================================

    def _partition_nodes(self):
        """Classify nodes as active (connected to inductor/JJ) or passive."""
        active, passive = [], []
        
        for node in self.circuit_graph.vertices:
            # Ground node is neither active nor passive
            if node.label == "gnd":
                continue
            
            _, inductive_degree = node._get_degree()
            
            # Active
            if inductive_degree > 0:
                active.append(node)
            # Passive
            else:
                passive.append(node)
                
        return active, passive

    # =====================================================================
    #   Matrix Construction
    # =====================================================================

    def _build_matrices(self):
        """
        Build the reduced (ground row/column removed) capacitance and
        inverse-inductance matrices using the graph-Laplacian approach.
        """
        # print(f"P: {self.P}")
        capacitance_matrix        = np.zeros((self.P, self.P))
        inverse_inductance_matrix = np.zeros((self.P, self.P))

        for branch in self.circuit_graph.edges:
            node1_label = branch.nodes[0].label
            node2_label = branch.nodes[1].label
            i = self.labels.index(node1_label)
            j = self.labels.index(node2_label)

            if isinstance(branch, Capacitor):
                capacitance_matrix[i][j] -= branch.C
                capacitance_matrix[j][i] -= branch.C
            elif isinstance(branch, Inductor):
                inverse_inductance_matrix[i][j] -= 1 / branch.L
                inverse_inductance_matrix[j][i] -= 1 / branch.L

        for i in range(self.P):
            capacitance_matrix[i][i]        = -np.sum(capacitance_matrix[i])
            inverse_inductance_matrix[j][j] = -np.sum(inverse_inductance_matrix[j])

        # Remove the row/col corresponding to gnd node
        capacitance_matrix = np.delete(np.delete(capacitance_matrix, 0, axis=0), 0, axis=1)
        inverse_inductance_matrix = np.delete(np.delete(inverse_inductance_matrix, 0, axis=0), 0, axis=1)
        
        return capacitance_matrix, inverse_inductance_matrix

    # =====================================================================
    #   Connectivity Display
    # =====================================================================

    def connectivity(self):
        s = "--- Circuit Connectivity ---\n"
        for node in self.circuit_graph.vertices:
            s += f"Node '{node.label}':\n"
            for branch in node.branches:
                other = [n for n in branch.nodes if n != node]
                conn = f"connected to {other[0].label}" if other else "grounded"
                if isinstance(branch, Capacitor):
                    s += f"  - Capacitor ({branch.C:.2e}) {conn}\n"
                elif isinstance(branch, Inductor):
                    s += f"  - Inductor ({branch.L:.2e}) {conn}\n"
                else:
                    s += f"  - Josephson Element ({branch.EJ:.2e}) {conn}\n"
        return s

    def __repr__(self):
        return self.connectivity()