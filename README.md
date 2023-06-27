# MyChemicalRomanceGNN
MDN GNN Miniproject

Molecules in chemistry are structured by atoms forming bonds with other atoms. Using a CNN type of network to predict what type of chemical a compound is and its effects is quite archaic, as it does not use the best information it can from the types of bonds, types of atoms, rather doing it all visually.

Graph neural networks are a relatively newer type of neural network that can work on graph-structured data and perform “convolutions” on graphs similar to images. These go through similar layers which ultimately end up in creating embedding representations that get pooled together into a final output which hopefully contains aggregated information about what we are interested in.

The project involves training a custom GNN model to accurately classify molecules as BACE enzyme inhibitors using node, edge, and graph level attributes. This will involve us utilizing the BACE dataset that can be found at https://moleculenet.org/datasets-1, which stores molecules in the SMILE format that can be processed using RDKit. This dataset also contains molecule-level attributes, which can also be utilized in the classification task.

Further documentation can be found on the Notion page (https://www.notion.so/Team-Homepage-fc9eb1065bca41928a0e12a5d4dacd10), please create a free Notion account with your student email and request access in the chat.
