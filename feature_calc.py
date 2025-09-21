from rdkit import Chem
import numpy as np
import os
import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch_geometric.nn.models import GAE
from torch_geometric.nn import global_mean_pool, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import joblib
import warnings

warnings.filterwarnings("ignore")

def calc_ecfp(smiles, col_prefix = "ecfp"):
    '''
    Calculating ECFP fingerprints from SMILES strings.
    
    '''
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    ecfp_list = []
    for mol in tqdm(mols, desc=f"Calculating {col_prefix} fingerprints"):
        if mol is None:
            ecfp_list.append(np.zeros(1024))
        else:
            ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024).GetCountFingerprintAsNumPy(mol)
            ecfp_list.append(ecfp)

    ecfp_array = np.array(ecfp_list)
    ecfp_df = pd.DataFrame(ecfp_array, columns=[f"{col_prefix}_{i}" for i in range(ecfp_array.shape[1])])
    return ecfp_df

def calc_maccs(smiles, col_prefix = "maccs"):
    '''
    Calculating MACCS keys from SMILES strings.
    
    '''
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    maccs_list = []
    for mol in tqdm(mols, desc=f"Calculating {col_prefix}"):
        if mol is None:
            maccs_list.append(np.zeros(167))
        else:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_list.append(np.array(maccs))

    maccs_array = np.array(maccs_list)
    maccs_df = pd.DataFrame(maccs_array, columns=[f"{col_prefix}_{i}" for i in range(maccs_array.shape[1])])
    return maccs_df

def calc_molecular_descriptors(smiles, col_prefix="mol_desc"):
    '''
    Calculating molecular descriptors using RDKit.
    
    '''
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    desc_names = [desc[0] for desc in Descriptors.descList]
    desc_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    mol_desc = []
    for mol in tqdm(mols, desc=f"Calculating {col_prefix} descriptors"):
        if mol is not None:
            mol = Chem.AddHs(mol)
            descs = desc_calculator.CalcDescriptors(mol)
        else:
            descs = [np.nan] * len(desc_names)
        mol_desc.append(descs)

    mol_desc_df = pd.DataFrame(mol_desc, columns=[f'{col_prefix}_{name}' for name in desc_names])
    return mol_desc_df

def featurize_ChemBERTa(smiles_list, padding=True, col_prefix="ChemBERTa"):
    embeddings_cls = np.zeros((len(smiles_list), 600))
    embeddings_mean = np.zeros((len(smiles_list), 600))
    chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list, desc=f"Calculating {col_prefix}")):
            encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
            model_output = chemberta(**encoded_input)
            
            embedding = model_output[0][::,0,::]
            # normalize the embedding
            embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
            embeddings_cls[i] = embedding
            
            embedding = torch.mean(model_output[0],1)
            embeddings_mean[i] = embedding

    embeddings_cls_df = pd.DataFrame(embeddings_cls, columns=[f"{col_prefix}_cls_{i}" for i in range(embeddings_cls.shape[1])])
    embeddings_mean_df = pd.DataFrame(embeddings_mean, columns=[f"{col_prefix}_mean_{i}" for i in range(embeddings_mean.shape[1])])

    embeddings = pd.concat([embeddings_cls_df, embeddings_mean_df], axis=1)
    return embeddings

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    edge_index = []
    node_features = []
    edge_features = []
    edge_feat_dim = 3  # bond type, is_conjugated, is_in_ring

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = bond.GetIsConjugated()
        is_in_ring = bond.IsInRing()

        edge_feat = [bond_type, float(is_conjugated), float(is_in_ring)]
        # double append for undirected graph
        edge_index.append((i, j))
        edge_index.append((j, i))
        edge_features.append(edge_feat)
        edge_features.append(edge_feat)

    for atom in mol.GetAtoms():
        atom_type = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        formal_charge = atom.GetFormalCharge()
        hybridization = int(atom.GetHybridization())
        degree = atom.GetDegree()
        valence = atom.GetTotalValence()
        node_features.append([atom_type, aromatic, formal_charge, hybridization, degree, valence])

    x = torch.tensor(node_features, dtype=torch.float)

    if len(edge_index) == 0:
        # Add self-loop for single atom
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, edge_feat_dim))
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4):
        super().__init__()
        dims = [in_channels] + [hidden_channels]*(num_layers-1) + [out_channels]
        self.convs = nn.ModuleList()
        for l in range(num_layers):
            in_c = dims[l]
            out_c = dims[l+1]
            # multi-head, then concat -> set concat=True except last where we want out_channels
            if l < num_layers - 1:
                self.convs.append(GATConv(in_c, out_c // heads, heads=heads, concat=True))
            else:
                self.convs.append(GATConv(in_c, out_c, heads=1, concat=True))
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x  # node embeddings
    

def calc_graph_embeddings(smiles_list, batch_size=32, col_prefix="graph", retrain=False):
    """
    Calculate graph embeddings
    
    Parameters:
    - smiles_list: List of SMILES strings.
    - batch_size: Batch size for processing.
    
    Returns:
    - embeddings: DataFrame with graph embeddings.
    """

    dataset = [mol_to_graph(smiles) for smiles in smiles_list]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = dataset[0].num_features
    hidden_channels = 128
    out_channels = 64
    epochs = 20
    num_layers = 3
    lr = 1e-3

    if not retrain and os.path.exists(f'./processed_datasets/preproc_objs/gat_encoder_{col_prefix}.pth'):
        print("loading existing GAT encoder model...")
        encoder = GATEncoder(in_channels, hidden_channels, out_channels, num_layers=num_layers).to(device)
        model = GAE(encoder).to(device)
        model.load_state_dict(torch.load(f'./processed_datasets/preproc_objs/gat_encoder_{col_prefix}.pth', map_location=device))
        model.eval()

        out_graph_embeds = []
        with torch.no_grad():
            for data in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
                data = data.to(device)
                z = model.encode(data.x, data.edge_index)          # [num_nodes_batch, out_channels]
                pooled = global_mean_pool(z, data.batch)           # [batch_graphs, out_channels]
                out_graph_embeds.append(pooled.cpu().numpy())

        embeddings = np.vstack(out_graph_embeds)
        return pd.DataFrame(embeddings, columns=[f'{col_prefix}_{i}' for i in range(embeddings.shape[1])])

    encoder = GATEncoder(in_channels, hidden_channels, out_channels, num_layers=num_layers).to(device)
    model = GAE(encoder).to(device)  # uses InnerProductDecoder by default

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # unsupervised reconstruction training
    model.train()
    print("training GAT encoder...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)  # node embeddings
            loss = model.recon_loss(z, data.edge_index)  # binary cross-entropy on observed edges
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        # print or log if needed:
        # print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(loader):.4f}")

    # Export graph-level embeddings (mean-pooled node embeddings per graph)
    model.eval()

    out_graph_embeds = []
    with torch.no_grad():
        for data in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)          # [num_nodes_batch, out_channels]
            pooled = global_mean_pool(z, data.batch)           # [batch_graphs, out_channels]
            out_graph_embeds.append(pooled.cpu().numpy())

    # save trained model for inference time
    torch.save(model.state_dict(), f'./processed_datasets/preproc_objs/gat_encoder_{col_prefix}.pth')

    embeddings = np.vstack(out_graph_embeds)

    return pd.DataFrame(embeddings, columns=[f'{col_prefix}_{i}' for i in range(embeddings.shape[1])])

def main():
    dataset_dir = "./processed_datasets"
    dataset_organics = pd.read_csv(os.path.join(dataset_dir, "organic_ions.csv"))

    dataset_type = 'organic'  # Change to 'all' for all ions
    if dataset_type == 'organic':
        output_dir = os.path.join(dataset_dir, "organic")
        os.makedirs(output_dir, exist_ok=True)

    elif dataset_type == 'all':
        output_dir = os.path.join(dataset_dir, "all")
        os.makedirs(output_dir, exist_ok=True)
    
    anions_smiles_list = dataset_organics["Anion"].tolist()
    cations_smiles_list = dataset_organics["Cation"].tolist()
    
    graph_anion = calc_graph_embeddings(anions_smiles_list, col_prefix="graph", retrain=True)
    graph_cation = calc_graph_embeddings(cations_smiles_list, col_prefix="graph", retrain=True)

    ecfp_anion = calc_ecfp(anions_smiles_list, col_prefix="ecfp")
    ecfp_cation = calc_ecfp(cations_smiles_list, col_prefix="ecfp")

    maccs_anion = calc_maccs(anions_smiles_list, col_prefix="maccs")
    maccs_cation = calc_maccs(cations_smiles_list, col_prefix="maccs")

    mol_desc_anion = calc_molecular_descriptors(anions_smiles_list, col_prefix="mol_desc")
    mol_desc_cation = calc_molecular_descriptors(cations_smiles_list, col_prefix="mol_desc")

    chemberta_anion = featurize_ChemBERTa(anions_smiles_list, col_prefix="ChemBERTa")
    chemberta_cation = featurize_ChemBERTa(cations_smiles_list, col_prefix="ChemBERTa")

    # Save all features to CSV files
    ecfp_anion.to_csv(os.path.join(output_dir,f"ecfp_anions_{dataset_type}.csv"), index=False)
    ecfp_cation.to_csv(os.path.join(output_dir, f"ecfp_cations_{dataset_type}.csv"), index=False)

    maccs_anion.to_csv(os.path.join(output_dir, f"anion_maccs_descriptors_{dataset_type}.csv"), index=False)
    maccs_cation.to_csv(os.path.join(output_dir, f"cation_maccs_descriptors_{dataset_type}.csv"), index=False)

    mol_desc_anion.to_csv(os.path.join(output_dir, f"anion_molecular_descriptors_{dataset_type}.csv"), index=False)
    mol_desc_cation.to_csv(os.path.join(output_dir, f"cation_molecular_descriptors_{dataset_type}.csv"), index=False)

    chemberta_anion.to_csv(os.path.join(output_dir, f"anion_chemberta_embeddings_{dataset_type}.csv"), index=False)
    chemberta_cation.to_csv(os.path.join(output_dir, f"cation_chemberta_embeddings_{dataset_type}.csv"), index=False)

    graph_anion.to_csv(os.path.join(output_dir, f"gat_embeddings_anions_{dataset_type}.csv"), index=False)
    graph_cation.to_csv(os.path.join(output_dir, f"gat_embeddings_cations_{dataset_type}.csv"), index=False)

    
    print("Feature extraction completed and saved to CSV files.")

if __name__ == "__main__":
    main()