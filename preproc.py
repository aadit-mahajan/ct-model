# fixed_feature_pipeline.py
# Merged + corrected version of your feature calculation pipeline.

from rdkit import Chem
import numpy as np
import os
import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
import torch
from torch_geometric.nn.models import GAE
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle as pkl
import joblib
from feature_calc import GATEncoder
from feature_eng import KBinsDiscretizerDF, ToDataFrame, KMeansFeatures

# ---- Globals to avoid re-loading heavy models repeatedly ----
# NOTE: set model_name to None if you don't want ChemBERTa embeddings
CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
_CHEMBERTA_MODEL = None
_CHEMBERTA_TOKENIZER = None
_CHEMBERTA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
#     tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

def init_chemberta(model_name=CHEMBERTA_MODEL_NAME):
    """Load ChemBERTa base model + tokenizer once (global)."""
    global _CHEMBERTA_MODEL, _CHEMBERTA_TOKENIZER
    if model_name is None:
        return None, None
    if _CHEMBERTA_MODEL is None or _CHEMBERTA_TOKENIZER is None:
        _CHEMBERTA_MODEL = AutoModelForMaskedLM.from_pretrained(model_name).to(_CHEMBERTA_DEVICE)
        _CHEMBERTA_MODEL.eval()
        _CHEMBERTA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _CHEMBERTA_MODEL, _CHEMBERTA_TOKENIZER

# ------------------- Feature calculations --------------------

def calculate_ecfp(smiles, radius=3, nBits=1024):
    """
    Calculate ECFP (Morgan fingerprint) as a numpy vector.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    arr = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits).GetCountFingerprintAsNumPy(mol)
    cols = [f'ecfp_{i}' for i in range(len(arr))]
    return pd.DataFrame([arr], columns=cols)

def calculate_maccs(smiles):
    """
    Calculate MACCS keys as numpy array.
    """
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    maccs_vect = MACCSkeys.GenMACCSKeys(mol)  # Explicit Bit Vector
    arr = np.array(list(map(int, maccs_vect.ToBitString())))  # convert to int array
    cols = [f'maccs_{i}' for i in range(len(arr))]
    return pd.DataFrame([arr], columns=cols)

def calculate_molecular_descriptors(smiles):
    """
    Calculate molecular descriptors using RDKit. Safe-wrapped so that if a descriptor raises
    for a given molecule we catch and set NaN for that descriptor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    try:
        descriptors = calculator.CalcDescriptors(mol)
    except Exception as e:
        # Try computing descriptors one-by-one to identify failing descriptors and set NaN for them
        descriptors = []
        for name in descriptor_names:
            try:
                val = getattr(Descriptors, name)(mol)
            except Exception:
                val = np.nan
            descriptors.append(val)
    cols = [f'mol_desc_{name}' for name in descriptor_names]
    return pd.DataFrame([descriptors], columns=cols)

def calculate_chemberta(smiles, padding=True):
    embeddings_cls = None
    embeddings_mean = None
    chemberta = _CHEMBERTA_MODEL
    tokenizer = _CHEMBERTA_TOKENIZER

    with torch.no_grad():
        encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
        model_output = chemberta(**encoded_input)
        
        embedding = model_output[0][::,0,::]
        # normalize the embedding
        embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
        embeddings_cls = embedding
        
        embedding = torch.mean(model_output[0],1)
        embeddings_mean = embedding

    embeddings_cls_df = pd.DataFrame(embeddings_cls, columns=[f"ChemBERTa_cls_{i}" for i in range(embeddings_cls.shape[1])])
    embeddings_mean_df = pd.DataFrame(embeddings_mean, columns=[f"ChemBERTa_mean_{i}" for i in range(embeddings_mean.shape[1])])

    embeddings = pd.concat([embeddings_cls_df, embeddings_mean_df], axis=1)
    return embeddings

def mol_to_graph(smiles):
    """
    Convert RDKit Mol -> torch_geometric.data.Data
    Node features are numeric floats. Edge index is 2xE long tensor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    node_features = []
    edge_index = []
    edge_features = []

    # Node features
    for atom in mol.GetAtoms():
        atom_type = atom.GetAtomicNum()
        aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        formal_charge = float(atom.GetFormalCharge())
        hybridization = float(int(atom.GetHybridization())) if atom.GetHybridization() is not None else 0.0
        degree = float(atom.GetDegree())
        valence = float(atom.GetTotalValence())
        node_features.append([atom_type, aromatic, formal_charge, hybridization, degree, valence])

    # Edge features (bond info) and undirected edges
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = float(bond.GetBondTypeAsDouble())
        is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
        is_in_ring = 1.0 if bond.IsInRing() else 0.0
        edge_feat = [bond_type, is_conjugated, is_in_ring]

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_features.append(edge_feat)
        edge_features.append(edge_feat)

    x = torch.tensor(node_features, dtype=torch.float) if len(node_features) > 0 else torch.zeros((0, 6), dtype=torch.float)

    if len(edge_index) == 0:
        # If no bonds (e.g., single atom), add self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, E]
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def calc_graph_embeddings(smiles_list, batch_size=32, col_prefix="graph", retrain=False):
    """
    Calculate graph embeddings
    
    Parameters:
    - smiles_list: List of SMILES strings.
    - batch_size: Batch size for processing.
    
    Returns:
    - embeddings: DataFrame with graph embeddings.
    """
    for smiles in smiles_list:
        print(smiles)
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

# ------------ Preprocessing helpers (unchanged but more robust) ------------

def remove_filtered_cols(embedding, embed_name=""):
    """
    Attempt to load dropped_cols_{embed_name}.pkl and drop columns.
    If file missing, return embedding unchanged.
    """
    preproc_objs_dir = './processed_datasets/preproc_objs/'
    fname = os.path.join(preproc_objs_dir, f'dropped_cols_{embed_name}.pkl')
    if os.path.exists(fname):
        dropped_cols = pkl.load(open(fname, 'rb'))
        # handle both list-of-names or boolean mask
        try:
            return embedding.drop(columns=dropped_cols, errors='ignore')
        except Exception:
            return embedding
    else:
        return embedding

def filter_and_pca_transform(ecfp, maccs, desc, chemberta, graph_emb, ion_type, verbose=False):
    """
    Load PCA objects + dropped column lists depending on ion_type.
    Returns transformed arrays (1D each).

    returns the feature set only for the specified ion type (no default)
    """

    preprocessed_model_path = './processed_datasets/preproc_objs/'

    # safe helper to load object or raise understandable error
    def _load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required preprocessing object not found: {path}")
        return pkl.load(open(path, 'rb'))

    def _get_pca_objs(suffix):
        pca_ecfp = _load(os.path.join(preprocessed_model_path, f'pca_ecfp{suffix}_comp.pkl'))
        pca_maccs = _load(os.path.join(preprocessed_model_path, f'pca_maccs{suffix}_comp.pkl'))
        pca_desc = _load(os.path.join(preprocessed_model_path, f'pca_molecular{suffix}_comp.pkl'))
        pca_chemberta = _load(os.path.join(preprocessed_model_path, f'pca_chemberta{suffix}_comp.pkl'))
        pca_graph = _load(os.path.join(preprocessed_model_path, f'pca_graph{suffix}_comp.pkl'))

        return (pca_ecfp, pca_maccs, pca_desc, pca_chemberta, pca_graph)
    
    def _get_dropped_cols(suffix):
        cols_to_remove_ecfp = _load(os.path.join(preprocessed_model_path, f'dropped_cols_ecfp{suffix}.pkl'))
        cols_to_remove_maccs = _load(os.path.join(preprocessed_model_path, f'dropped_cols_maccs{suffix}.pkl'))
        cols_to_remove_desc = _load(os.path.join(preprocessed_model_path, f'dropped_cols_molecular{suffix}.pkl'))
        cols_to_remove_chemberta = _load(os.path.join(preprocessed_model_path, f'dropped_cols_chemberta{suffix}.pkl'))
        cols_to_remove_graph = _load(os.path.join(preprocessed_model_path, f'dropped_cols_graph{suffix}.pkl'))

        return (cols_to_remove_ecfp, cols_to_remove_maccs, cols_to_remove_desc,
                cols_to_remove_chemberta, cols_to_remove_graph)

    if ion_type == 'anion':
        suffix = '_a'
        pca_ecfp, pca_maccs, pca_desc, pca_chemberta, pca_graph = _get_pca_objs(suffix)
        del_cols_ecfp, del_cols_maccs, del_cols_desc, del_cols_chemberta, del_cols_graph = _get_dropped_cols(suffix)
    if ion_type == 'cation':
        suffix = '_c'
        pca_ecfp, pca_maccs, pca_desc, pca_chemberta, pca_graph = _get_pca_objs(suffix)
        del_cols_ecfp, del_cols_maccs, del_cols_desc, del_cols_chemberta, del_cols_graph = _get_dropped_cols(suffix)

    if verbose: 
        print(f'Dropped columns for {ion_type}:', 
              len(del_cols_ecfp), len(del_cols_maccs), len(del_cols_desc), 
              len(del_cols_chemberta), len(del_cols_graph))
        print("-" * 40, '\n')

    # drop columns
    def _drop_cols(df_like, drop_cols, col_prefix=""):
        if isinstance(df_like, pd.DataFrame):
            df_dropped = df_like.drop(columns=drop_cols, errors='ignore')
        else:
            raise TypeError("Expected a pandas DataFrame.")

        return df_dropped

    if verbose:
        print('ecfp cols', ecfp.shape)
        print('maccs cols', maccs.shape)
        print('desc cols', desc.shape)
        print('chemberta cols', chemberta.shape)
        print('graph cols', graph_emb.shape)

    ecfp_arr = _drop_cols(ecfp, del_cols_ecfp, col_prefix="ecfp")
    maccs_arr = _drop_cols(maccs, del_cols_maccs, col_prefix="maccs")
    desc_arr = _drop_cols(desc, del_cols_desc, col_prefix="desc")
    chemberta_arr = _drop_cols(chemberta, del_cols_chemberta, col_prefix="chemberta")
    graph_arr = _drop_cols(graph_emb, del_cols_graph, col_prefix="graph")

    if verbose:
        print('ecfp cols', ecfp_arr.shape)
        print('maccs cols', maccs_arr.shape)
        print('desc cols', desc_arr.shape)
        print('chemberta cols', chemberta_arr.shape)
        print('graph cols', graph_arr.shape)

    # compress using pca
    ecfp_pca = pca_ecfp.transform(ecfp_arr)
    maccs_pca = pca_maccs.transform(maccs_arr)
    desc_pca = pca_desc.transform(desc_arr)
    chemberta_pca = pca_chemberta.transform(chemberta_arr)
    graph_pca = pca_graph.transform(graph_arr)

    if ion_type == 'anion':
        ecfp_pca = pd.DataFrame(ecfp_pca, columns=[f'PCA_ecfp_a_{i}' for i in range(ecfp_pca.shape[1])])
        maccs_pca = pd.DataFrame(maccs_pca, columns=[f'PCA_maccs_a_{i}' for i in range(maccs_pca.shape[1])])
        desc_pca = pd.DataFrame(desc_pca, columns=[f'PCA_molecular_a_{i}' for i in range(desc_pca.shape[1])])
        chemberta_pca = pd.DataFrame(chemberta_pca, columns=[f'PCA_chemberta_a_{i}' for i in range(chemberta_pca.shape[1])])
        graph_pca = pd.DataFrame(graph_pca, columns=[f'PCA_graph_a_{i}' for i in range(graph_pca.shape[1])])

    if ion_type == 'cation':
        ecfp_pca = pd.DataFrame(ecfp_pca, columns=[f'PCA_ecfp_c_{i}' for i in range(ecfp_pca.shape[1])])
        maccs_pca = pd.DataFrame(maccs_pca, columns=[f'PCA_maccs_c_{i}' for i in range(maccs_pca.shape[1])])
        desc_pca = pd.DataFrame(desc_pca, columns=[f'PCA_molecular_c_{i}' for i in range(desc_pca.shape[1])])
        chemberta_pca = pd.DataFrame(chemberta_pca, columns=[f'PCA_chemberta_c_{i}' for i in range(chemberta_pca.shape[1])])
        graph_pca = pd.DataFrame(graph_pca, columns=[f'PCA_graph_c_{i}' for i in range(graph_pca.shape[1])])

    if verbose:
        print("After PCA:")
        print('ecfp cols', ecfp_pca.shape)
        print('maccs cols', maccs_pca.shape)
        print('desc cols', desc_pca.shape)
        print('chemberta cols', chemberta_pca.shape)
        print('graph cols', graph_pca.shape)

    return ecfp_pca, maccs_pca, desc_pca, chemberta_pca, graph_pca

def scale_features(features, scaler_path):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    return scaler.transform(features)

def load_and_transform(data, load_path="feature_pipeline.pkl"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"feature pipeline not found: {load_path}")
    obj = joblib.load(load_path)
    pipeline = obj["pipeline"]

    # Ensure columns match what the pipeline expects
    data = data.copy()
    data.columns = data.columns.astype(str)
    if hasattr(pipeline, "feature_names_in_"):
        train_cols = pipeline.feature_names_in_
        data_selected = data.reindex(columns=train_cols, fill_value=0)

    transformed_data = pipeline.transform(data_selected)
    print("Transformed data shape:", transformed_data.shape)
    print("Original data shape:", data.shape)
    # Otherwise, try to get column names from the pipeline
    if isinstance(transformed_data, np.ndarray):
        # Try to get column names from transformers
        col_names = []

        # mostly fallbacks if transformers don't have get_feature_names_out
        X_selected = pipeline.named_steps['select_k'].transform(data)
        for name, trans in pipeline.named_steps['features'].transformer_list:
            if hasattr(trans, 'named_steps'):
                last_step = list(trans.named_steps.values())[-1]
                if hasattr(last_step, 'get_feature_names_out'):
                    col_names.extend(last_step.get_feature_names_out())
                else:
                    col_names.extend([f"{name}_{i}" for i in range(trans.transform(X_selected).shape[1])])
            else:
                col_names.extend([f"{name}_{i}" for i in range(trans.transform(X_selected).shape[1])])
        if len(col_names) != transformed_data.shape[1]:
            col_names = [f"feat_{i}" for i in range(transformed_data.shape[1])]
        transformed_data = pd.DataFrame(transformed_data, columns=col_names, index=data.index)

    # concat to original data
    transformed_data = pd.concat([data.reset_index(drop=True), transformed_data.reset_index(drop=True)], axis=1)

    return transformed_data

# ------------------ High-level generate_features ---------------------

def generate_features(cation_smiles, anion_smiles):
    """
    Generate the final features DataFrame for a given SMILES.
    embed_type in {'cation', 'anion', 'full'} selects which PCA/scaler to apply.
    """
    # compute base features
    ecfp_c = calculate_ecfp(cation_smiles)
    maccs_c = calculate_maccs(cation_smiles)
    desc_c = calculate_molecular_descriptors(cation_smiles)
    chemberta_c = calculate_chemberta(cation_smiles)
    # graph_data_c = mol_to_graph(cation_smiles)
    graph_emb_c = calc_graph_embeddings([cation_smiles])

    ecfp_a = calculate_ecfp(anion_smiles)
    maccs_a = calculate_maccs(anion_smiles)
    desc_a = calculate_molecular_descriptors(anion_smiles)
    chemberta_a = calculate_chemberta(anion_smiles)
    # graph_data_a = mol_to_graph(anion_smiles)
    graph_emb_a = calc_graph_embeddings([anion_smiles])

    # raise on invalid SMILES
    if ecfp_c is None or maccs_c is None or desc_c is None:
        raise ValueError(f"Invalid SMILES string or failed featurization for: {cation_smiles}")

    if ecfp_a is None or maccs_a is None or desc_a is None:
        raise ValueError(f"Invalid SMILES string or failed featurization for: {anion_smiles}")

    # remove filtered columns (if drop lists exist)
    ecfp_c = remove_filtered_cols(ecfp_c, embed_name='ecfp')
    maccs_c = remove_filtered_cols(maccs_c, embed_name='maccs')
    desc_c = remove_filtered_cols(desc_c, embed_name='molecular')
    chemberta_c = remove_filtered_cols(chemberta_c, embed_name='chemberta')
    graph_emb_c = remove_filtered_cols(graph_emb_c, embed_name='graph')

    ecfp_a = remove_filtered_cols(ecfp_a, embed_name='ecfp')
    maccs_a = remove_filtered_cols(maccs_a, embed_name='maccs')
    desc_a = remove_filtered_cols(desc_a, embed_name='molecular')
    chemberta_a = remove_filtered_cols(chemberta_a, embed_name='chemberta')
    graph_emb_a = remove_filtered_cols(graph_emb_a, embed_name='graph')
    # debug print shapes
    # print(f"ECFP shape: {ecfp_c.shape}, MACCS shape: {maccs_c.shape}, Descriptors shape: {desc_c.shape}, ChemBERTa shape: {chemberta_c.shape}, Graph Embeddings shape: {graph_emb_c.shape}")

    # PCA + removal + scaling
    ecfp_pca_c, maccs_pca_c, desc_pca_c, chemberta_pca_c, graph_pca_c = filter_and_pca_transform(
        ecfp_c, maccs_c, desc_c, chemberta_c, graph_emb_c, ion_type='cation'
    )

    ecfp_pca_a, maccs_pca_a, desc_pca_a, chemberta_pca_a, graph_pca_a = filter_and_pca_transform(
        ecfp_a, maccs_a, desc_a, chemberta_a, graph_emb_a, ion_type='anion'
    )

    # order of the features in the dataset: chemberta, ecfp, maccs, desc, graph
    # combine all features in dataframe format itself
    features_df = pd.concat([
        pd.DataFrame(chemberta_pca_c),
        pd.DataFrame(ecfp_pca_c),
        pd.DataFrame(maccs_pca_c),
        pd.DataFrame(desc_pca_c),
        pd.DataFrame(graph_pca_c),
        pd.DataFrame(chemberta_pca_a),
        pd.DataFrame(ecfp_pca_a),
        pd.DataFrame(maccs_pca_a),
        pd.DataFrame(desc_pca_a),
        pd.DataFrame(graph_pca_a),
    ], axis=1)
    print("Combined features shape before final pipeline:", features_df.shape)
    # Load and transform to select top features -> pipeline returns numpy
    final_df = load_and_transform(features_df, load_path='./processed_datasets/preproc_objs/feature_pipeline.pkl')

    return final_df

def featurize_smiles_list(cation_smiles_list, anion_smiles_list):
    all_features = pd.DataFrame()
    
    if len(cation_smiles_list) != len(anion_smiles_list):
        raise ValueError("Cation and anion lists must be of the same length.")

    for idx in tqdm(range(len(cation_smiles_list)), desc="Featurizing SMILES"):
        try:
            features_df = generate_features(cation_smiles_list[idx], anion_smiles_list[idx])
            all_features = pd.concat([all_features, features_df])
        except Exception as e:
            print(f"Error processing SMILES {cation_smiles_list[idx]}: {e}")

    if not all_features.empty:
        return all_features
    else:
        return pd.DataFrame()

# ---- if running as script ----
if __name__ == "__main__":
    # initialize ChemBERTa once (if available)
    try:
        init_chemberta()
    except Exception as e:
        print("Warning: failed to load ChemBERTa (will continue without it).", e)
        CHEMBERTA_MODEL_NAME = None

    test_cation = 'CCCCCCCCCC[n+]1ccc(C(=O)N/N=C/c2ccc[n+](CCCCCCCCCC)c2)cc1'
    test_anion = 'O=C([O-])C(F)(F)F'
    target = 0.02001 #target -log ct value for this pair
    features = featurize_smiles_list([test_cation], [test_anion])
    print("Result shape:", features.shape)
    if not features.empty:
        print(features.head())
