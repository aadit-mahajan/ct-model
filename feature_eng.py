import pandas as pd
import numpy as np
import os
from umap import UMAP
from sklearn.preprocessing import StandardScaler

dataset_dir = "ct-model/cyto_datasets_2025-04-01/processed_datasets"

def load_chemberta_embeddings():
    embeddings_df = pd.read_csv(os.path.join(dataset_dir, "chemberta_embeddings.csv"))
    return embeddings_df

def load_ecfp_fingerprints():
    ecfp_df = pd.read_csv(os.path.join(dataset_dir, "ecfp_fingerprints.csv"))
    return ecfp_df

def load_maccs_descriptors():
    maccs_df = pd.read_csv(os.path.join(dataset_dir, "maccs_descriptors.csv"))
    return maccs_df

def load_molecular_descriptors():
    mol_desc_df = pd.read_csv(os.path.join(dataset_dir, "molecular_descriptors.csv"))
    return mol_desc_df

def load_graph_embeddings():
    graph_df = pd.read_csv(os.path.join(dataset_dir, "graph_transformer_embeddings.csv"))
    return graph_df

def compress_embeddings(embeddings_df, embed_name, n_components):
    """
    Compress the embeddings using UMAP.
    
    Parameters:
    - embeddings_df: DataFrame containing the embeddings.
    - n_components: Number of dimensions to reduce to.
    
    Returns:
    - compressed_embeddings: DataFrame with compressed embeddings.
    """

    # print("Max value:", embeddings_df.max().max())
    print("Any infs:", np.isinf(embeddings_df.values).any())
    print("Any NaNs:", np.isnan(embeddings_df.values).any())

    umap = UMAP(n_components=n_components, n_jobs=-1)
    compressed_embeddings = umap.fit_transform(embeddings_df.astype(np.float32))
    return pd.DataFrame(compressed_embeddings, columns=[f'UMAP_{embed_name}_{i}' for i in range(n_components)])


def filter_embeddings(df, corr_threshold=0.9, verbose=False):
    original_shape = df.shape

    # Step 1: Remove inf/nan values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    # Step 2: Drop constant columns
    nunique = df.apply(pd.Series.nunique)
    constant_cols = nunique[nunique <= 1].index
    df = df.drop(columns=constant_cols)

    # Step 3: Drop columns with extreme magnitudes
    abs_max_vals = df.abs().max()
    unstable_cols = abs_max_vals[abs_max_vals > 1e6].index
    df = df.drop(columns=unstable_cols)

    # Step 4: Drop highly correlated columns
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    df = df.drop(columns=to_drop)

    # Step 5: Remove any remaining rows with inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Step 6: Normalize to avoid scale dominance
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

    if verbose:
        print(f"Original shape: {original_shape}")
        print(f"Removed constant columns: {len(constant_cols)}")
        print(f"Removed unstable magnitude columns: {len(unstable_cols)}")
        print(f"Removed highly correlated columns: {len(to_drop)}")
        print(f"Final shape: {df.shape}")
        print("-" * 40)

    return df

def generate_dataset():
    """
    Generate a dataset by loading and compressing all embeddings.
    
    Returns:
    - dataset: DataFrame containing all compressed embeddings.
    """

    chemberta_embeddings = load_chemberta_embeddings()
    ecfp_fingerprints = load_ecfp_fingerprints()
    maccs_descriptors = load_maccs_descriptors()
    molecular_descriptors = load_molecular_descriptors()
    graph_embeddings = load_graph_embeddings()

    # Clean the dataframes to remove NaN values
    chemberta_embeddings = filter_embeddings(chemberta_embeddings, corr_threshold=0.95, verbose=True)
    ecfp_fingerprints = filter_embeddings(ecfp_fingerprints, corr_threshold=0.95, verbose=True)
    maccs_descriptors = filter_embeddings(maccs_descriptors, corr_threshold=0.95, verbose=True)
    molecular_descriptors = filter_embeddings(molecular_descriptors, corr_threshold=0.95, verbose=True)
    graph_embeddings = filter_embeddings(graph_embeddings, corr_threshold=0.95, verbose=True)

    # compile an uncompressed dataset
    print("Compiling uncompressed dataset...")
    uncompressed_dataset = pd.concat([chemberta_embeddings, ecfp_fingerprints, maccs_descriptors, molecular_descriptors, graph_embeddings], axis=1)


    compressed_chemberta = compress_embeddings(chemberta_embeddings, n_components=64, embed_name='chemberta')
    print("chemberta embeddings done\n")
    compressed_ecfp = compress_embeddings(ecfp_fingerprints, n_components=16, embed_name='ecfp')
    print("ecfp fingerprints done\n")
    compressed_maccs = compress_embeddings(maccs_descriptors, n_components=64, embed_name='maccs')
    print("maccs descriptors done\n")
    compressed_molecular = compress_embeddings(molecular_descriptors, n_components=64, embed_name='molecular')
    print("molecular descriptors done\n")
    compressed_graph = compress_embeddings(graph_embeddings, n_components=16, embed_name='graph')
    print("graph embeddings done\n")

    dataset = pd.concat([compressed_chemberta, compressed_ecfp, compressed_maccs, compressed_molecular, compressed_graph], axis=1)

    # Add cytotoxicity values and SMILES strings
    cytotoxicity_data = pd.read_csv(os.path.join(dataset_dir, "cytotoxicity_data.csv"))

    # cytotoxicity value saved as target variable
    dataset['target'] = cytotoxicity_data['CC50/IC50/EC50, mM']
    uncompressed_dataset['target'] = cytotoxicity_data['CC50/IC50/EC50, mM']
    dataset['Canonical SMILES'] = cytotoxicity_data['Canonical SMILES']
    uncompressed_dataset['Canonical SMILES'] = cytotoxicity_data['Canonical SMILES']
    dataset['filename'] = cytotoxicity_data['filename']
    uncompressed_dataset['filename'] = cytotoxicity_data['filename']

    return dataset, uncompressed_dataset

def main():
    dataset, uncompressed_dataset = generate_dataset()
    # Save the final dataset
    dataset.to_csv(os.path.join(dataset_dir, "final_dataset.csv"), index=False)
    uncompressed_dataset.to_csv(os.path.join(dataset_dir, "uncompressed_dataset.csv"), index=False)

if __name__ == "__main__":
    main()
    print("Dataset generated and saved to processed_datasets/final_dataset.csv")