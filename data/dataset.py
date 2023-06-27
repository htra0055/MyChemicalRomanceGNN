import pandas as pd
from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
import numpy as np 
import os
from tqdm import tqdm
import random

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, filename = None, dataset_num = 0, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        # self.test = test # test/train split (bool)
        self.train_val_test = dataset_num
        if filename is not None:
            self.filename = filename # name of csv file containing BACE dataset (string)
        else:
            raise Exception
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ 
        
        If these files are found in raw_dir, processing is skipped.

        We will use three files to store our train, test and validation Data objects
        in memory. If these files are found in processed directory, super() will not 
        call the process() method.
        
        """
        return ['train.pt', 'test.pt', 'val.pt']

        # self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        # if self.test:
        #     return [f'data_test_{i}.pt' for i in list(self.data.index)] 
        # else:
        #     return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        '''
        train_
        '''
        self.data = pd.read_csv(self.raw_paths[0])
        data_list = []
        
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Get graph features
            mol_feats = self._get_graph_features(mol)
            
            mol_obj = Chem.MolFromSmiles(mol["mol"])
            # Get node features
            node_feats = self._get_node_features(mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features(mol_obj)
            # Get adjacency matrix info
            edge_index = self._get_adjacency_info(mol_obj)
            # Get labels info
            label = self._get_labels(mol["Class"])

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label, 
                        graph_feats = mol_feats)
                        
            data_list.extend(data)
        
        random.shuffle(data_list)
        print(data_list[0].__class__)
        num_mols = len(self.data.index)
        train = list(data_list[0:int(np.floor(num_mols*0.8))])
        test = list(data_list[int(np.floor(num_mols*0.8)):int(np.floor(num_mols*0.9))])
        val = list(data_list[int(np.floor(num_mols*0.9)):-1])
        
        data, slices = self.collate(data_list=train)
        torch.save((data, slices), self.processed_paths[0])

        data, slices = self.collate(data_list=test)
        torch.save((data, slices), self.processed_paths[1])

        data, slices = self.collate(data_list=val)
        torch.save((data, slices), self.processed_paths[2])

    def _get_node_features(self, mol):
        """ 
        Inputs:
        * mol (molecule object)

        Outputs:
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """ 
        Inputs:
        * mol (molecule object)
        
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """
        Inputs:
        * mol (molecule object)
        Output: 
        * edge_indices
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices
    
    def _get_graph_features(self,mol_row):
        '''
        input:
        - mol_row (csv row): csv row containing molecules data

        output: 
        - graph_feats: tensor containing the ordered graph features for given molecule
        [plc50, alogp, hba, hbd, psa, mw]
        '''
        graph_feats = []
        indices = [4,6,7,8,14,5]
        for index in indices:
            graph_feats.append(mol_row[index])
        return torch.tensor(graph_feats, dtype=torch.float)
    
    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    # def get(self, idx):
    #     # """ 
        # get retrieves a data set (train,test,val) given the data ID 
        # - Equivalent to __getitem__ in pytorch
        #     - Is not needed for PyG's InMemoryDataset
       
        # """
        
        # case self.dataset_num == 0: # If 
        #     data = torch.load(os.path.join(self.processed_dir, 
        #                          f'train.pt'))
        # elif self.train_val_test == 0:
        #     data = torch.load(os.path.join(self.processed_dir, 
        #                          f'data_{idx}.pt'))   
        # return data
    
#if __name__ == "main":
root_dir = os.getcwd() + r"/rawData"
print(root_dir)
#MyChemicalRomanceGNN/rawData
dataset = MoleculeDataset(root_dir, 'bace.csv')
