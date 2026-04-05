###############################################################################
#
#   Circuit.py
#
#   Builds a circuit graph from a user-defined topology and constructs the
#   capacitance and inductance matrices.
#
###############################################################################

import numpy as np
from Elements import *

###############################################################################
#
#   Circuit Class
#
###############################################################################

class Circuit:
    """
    Represents a superconducting circuit as a graph.
    Responsibilities: graph construction, capacitance/inductance matrices,
    and classical energy functions.
    """

    # =====================================================================
    #   Constructor
    # =====================================================================

    def __init__(self, graph_rep: dict):
        self.labels                                           = graph_rep["nodes"]
        self.circuit_graph                                    = Circuit._build_master_graph(graph_rep)
        self.capacitive_sub_graph, self.inductive_sub_graph   = self._build_sub_graphs()
        self.josephson_elements                               = [e for e in self.inductive_sub_graph.edges if isinstance(e, JosephsonElement)]
        self.active_nodes, self.passive_nodes                 = self._partition_nodes()
        self.N                                                = len(self.active_nodes)
        self.P                                                = len(self.active_nodes) + len(self.passive_nodes) + 1 # + 1 for ground
        self.capacitance_matrix, self.inv_inductance_matrix   = self._build_matrices()
        self.inv_capacitance_matrix                           = np.linalg.inv(self.capacitance_matrix)
        self.offset_dict                                      = graph_rep["external_flux"]
        self.omega_squared                                    = self._build_omega_squared()
        self.normal_modes_squared, self.normal_vecs_squared   = np.linalg.eig(self.omega_squared)

    # =====================================================================
    #   Graph Construction
    # =====================================================================

    @staticmethod
    def _build_master_graph(graph_rep: dict) -> Graph:
        """
        Build the full circuit graph from the user-supplied dictionary.
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
        capacitance_matrix        = np.zeros((self.P, self.P))
        inverse_inductance_matrix = np.zeros((self.P, self.P))

        for branch in self.circuit_graph.edges:
            node1_label = branch.nodes[0].label
            node2_label = branch.nodes[1].label
            i = 0 if node1_label == "gnd" else self.labels.index(node1_label) + 1 # + 1 to account for gnd being absent in labels
            j = 0 if node2_label == "gnd" else self.labels.index(node2_label) + 1

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
    #   Classical Mechanics Helpers
    # =====================================================================

    def _get_node_flux_dot(self, node_charge):
        return self.inv_capacitance_matrix @ node_charge

    def get_kinetic_energy(self, node_charge):
        node_flux_dot = self._get_node_flux_dot(node_charge)
        return (node_flux_dot.T @ self.capacitance_matrix @ node_flux_dot) / 2

    def get_potential_energy(self, node_flux: np.ndarray):
        """Total potential energy = linear (inductive) + nonlinear (Josephson)."""
        # Linear inductive term
        if self.inv_inductance_matrix.size > 0 and np.any(self.inv_inductance_matrix != 0):
            linear_term = (node_flux.T @ self.inv_inductance_matrix @ node_flux) / 2
        else:
            linear_term = 0

        # External flux correction
        ext_term = 0
        for (n1_label, n2_label), offset in self.offset_dict.items():
            j = self.labels.index(n1_label)
            k = self.labels.index(n2_label)
            inductance = -1 / self.inv_inductance_matrix[j][k]
            ext_term += ((node_flux[j] - node_flux[k]) * offset) / inductance

        # Josephson (nonlinear) term
        jj_term = 0
        for jj in self.josephson_elements:
            n1_label = jj.nodes[0].label
            n2_label = jj.nodes[1].label
            phi1 = node_flux[self.labels.index(n1_label)] if n1_label != "gnd" else 0
            phi2 = node_flux[self.labels.index(n2_label)] if n2_label != "gnd" else 0
            jj_term += -jj.EJ * np.cos((phi1 - phi2) / PHI_0)

        return linear_term + ext_term + jj_term

    def get_hamiltonian(self, node_flux, node_charge):
        return 0.5 * (node_charge.T @ self.inv_capacitance_matrix @ node_charge) + self.get_potential_energy(node_flux)

    def get_lagrangian(self, node_flux, node_charge):
        return self.get_kinetic_energy(node_charge) - self.get_potential_energy(node_flux)

    # =====================================================================
    #   Normal Mode Analysis
    # =====================================================================

    def _build_omega_squared(self):
        if np.any(self.inv_inductance_matrix != 0):
            return self.inv_capacitance_matrix @ self.inv_inductance_matrix
        return np.zeros((self.N, self.N))

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