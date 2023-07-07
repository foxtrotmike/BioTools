#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:18:00 2023
@author: Fayyaz Minhas


This code generates a pytorch geometric graph object for a given PDB file which can be used for training with graph neural networks
Each alpha carbon is considered as a node
Each node is represented by one hot encoding of its amino acid type
Each node is connected to upto 10 nearest neighbors that lie within 10 angstroms
Each edge has a single feature which is the distance
All ligands and other non-standard molecules are ignored
coords stores the location
The output is saved to a gml file which can be imported into cytoscape (see cytoscape repository) and visualized in 3d using the cy3D tool by setting the X,Y and Z location
Some parts of the code have been Generated with support from ChatGPT with the following prompt:
    
    Write a code snippet with explanation to convert a protein in a pdb file to a pytorch geometric graph object in which a node corresponds to a residue with its node features being the one hot encoding of its amino acid type and each node being connected to its neighbors within 10 angstroms.  Consider only alpha carbons for each amino acid. Discard any other atoms, ligands  or non-standard amino acids. 


data has the following attributes

data.x: Node feature matrix with shape [num_nodes, num_node_features]

data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

data.pos: Node position matrix with shape [num_nodes, num_dimensions] #we used coords instead

"""

import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities
from scipy.spatial import cKDTree
from Bio.PDB.Polypeptide import three_to_one
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
def writePyGGraph(G,ofname = 'temp.gml'):
    dict_coords = {'c'+str(i):G.coords[:,i] for i in range(G.coords.shape[1])}
    dict_feats = {'f'+str(i):G.x[:,i] for i in range(G.x.shape[1])}
    #import pdb;pdb.set_trace()
    dict_y = None# {'y'+str(i):G.y[:,i] for i in range(G.y.shape[1])}
    node_dict = {**dict_coords, **dict_feats}#,**dict_y
    d = Data(**node_dict,edge_index = G.edge_index, edge_attr = G.edge_attr)
    nG = to_networkx(d, node_attrs=list(node_dict.keys()))
    #nx.nx_pydot.write_dot(nG,'temp.dot')
    nx.write_gml(nG,ofname)
    
# input pdb File
pdb_file = "./1cll.pdb"


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
one_hot = {aa: torch.eye(len(amino_acids))[i] for i, aa in enumerate(amino_acids)}

# Parse the PDB file
parser = PDBParser()
structure = parser.get_structure("protein", pdb_file)

# Select alpha carbon atoms of standard amino acids
residues = []
for res in structure.get_residues():
    try:
        if three_to_one(res.get_resname()) in amino_acids:
            residues.append(res)
    except:
        continue

#residues = [res  if  in amino_acids]
alpha_carbons = [atom for res in residues for atom in res.get_atoms() if atom.get_name() == "CA"]

# Build a KDTree for efficient neighbor searching
positions = torch.tensor([atom.get_coord() for atom in alpha_carbons])
kdtree = cKDTree(positions.numpy())

# Find neighbors within 10 angstroms
distances, neighbors = kdtree.query(positions, k=9, distance_upper_bound=10.0)

# Create node features and edge indices
x = torch.stack([one_hot[three_to_one(res.get_resname())] for res in residues])
edge_index = []
edge_distance = []

for i, neighbor_indices in enumerate(neighbors):
    vidx = np.nonzero(neighbor_indices != len(residues))[0]
    valid_neighbors = neighbor_indices[vidx]  # Exclude invalid indices
    edge_index.extend([(i, n) for n in valid_neighbors])
    edge_distance.extend([distances[i, j] for j in vidx])

edge_features = torch.tensor(np.atleast_2d(edge_distance).T) #number of edges x 1 (single feature)
# Create the PyTorch Geometric Data object
data = Data(x=x, edge_index=torch.tensor(edge_index).t().contiguous(),coords=positions,edge_attr=torch.tensor(edge_distance))

# Print the resulting graph data output
print(data)
