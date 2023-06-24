# MyChemicalRomanceGNN
MDN GNN Miniproject

Molecules in chemistry are structured by atoms forming bonds with other atoms. Using a CNN type of network to predict what type of chemical a compound is and its effects is quite archaic, as it does not use the best information it can from the types of bonds, types of atoms, rather doing it all visually.

Graph neural networks are a relatively newer type of neural network that can work on graph-structured data and perform “convolutions” on graphs similar to images. These go through similar layers which ultimately end up in creating embedding representations that get pooled together into a final output which hopefully contains aggregated information about what we are interested in.

In this project, you will create a GNN model that is able to take in a proposed molecule and classify whether it can act as an HIV inhibitor. This decision can come down to type of atom, types of bonds (edges) etc.
