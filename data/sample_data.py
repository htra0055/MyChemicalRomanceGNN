### Load the Data ###
import pandas as pd
DATA_PATH = "/MyChemicalRomanceGNN/rawData/bace.csv"
raw_data = pd.read_csv(DATA_PATH)
print(raw_data.head(5)) # visualise the first 5 rows

### Info about Data ###
print(raw_data.shape)
print(raw_data['Class'].value_counts()) # 0 - Not a HIV inhibitor, 1 - Is a HIV inhibitor

### Sample Visualisation ###
from rdkit import Chem
from rdkit.Chem import Draw

sample_datapoints = raw_data['mol'][0:10].values # smile is the format they save their molecules
sample_mols = [Chem.MolFromSmiles(smiles) for smiles in sample_datapoints]
im_grid = Draw.MolsToGridImage(sample_mols, molsPerRow=5, subImgSize=(200,200))
im_grid.show()