import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures, KBinsDiscretizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pickle as pkl
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
import scipy.sparse as sp

dataset_dir = "./processed_datasets"
preproc_objs_dir = "./processed_datasets/preproc_objs"
os.makedirs(preproc_objs_dir, exist_ok=True)

def load_embeddings():
    return {
        "chemberta": (
            pd.read_csv(os.path.join(dataset_dir, "cation_chemberta_embeddings.csv")),
            pd.read_csv(os.path.join(dataset_dir, "anion_chemberta_embeddings.csv"))
        ),
        "ecfp": (
            pd.read_csv(os.path.join(dataset_dir, "ecfp_cations.csv")),
            pd.read_csv(os.path.join(dataset_dir, "ecfp_anions.csv"))
        ),
        "maccs": (
            pd.read_csv(os.path.join(dataset_dir, "cation_maccs_descriptors.csv")),
            pd.read_csv(os.path.join(dataset_dir, "anion_maccs_descriptors.csv"))
        ),
        "molecular": (
            pd.read_csv(os.path.join(dataset_dir, "cation_molecular_descriptors.csv")),
            pd.read_csv(os.path.join(dataset_dir, "anion_molecular_descriptors.csv"))
        ),
        "graph": (
            pd.read_csv(os.path.join(dataset_dir, "gat_embeddings_cations.csv")),
            pd.read_csv(os.path.join(dataset_dir, "gat_embeddings_anions.csv"))
        )
    }

def load_organic_embeddings():
    return {
        "chemberta": (
            pd.read_csv(os.path.join(dataset_dir, "organic/cation_chemberta_embeddings_organic.csv")),
            pd.read_csv(os.path.join(dataset_dir, "organic/anion_chemberta_embeddings_organic.csv"))
        ),
        "ecfp": (
            pd.read_csv(os.path.join(dataset_dir, "organic/ecfp_cations_organic.csv")),
            pd.read_csv(os.path.join(dataset_dir, "organic/ecfp_anions_organic.csv"))
        ),
        "maccs": (
            pd.read_csv(os.path.join(dataset_dir, "organic/cation_maccs_descriptors_organic.csv")),
            pd.read_csv(os.path.join(dataset_dir, "organic/anion_maccs_descriptors_organic.csv"))
        ),
        "molecular": (
            pd.read_csv(os.path.join(dataset_dir, "organic/cation_molecular_descriptors_organic.csv")),
            pd.read_csv(os.path.join(dataset_dir, "organic/anion_molecular_descriptors_organic.csv"))
        ),
        "graph": (
            pd.read_csv(os.path.join(dataset_dir, "organic/gat_embeddings_cations_organic.csv")),
            pd.read_csv(os.path.join(dataset_dir, "organic/gat_embeddings_anions_organic.csv"))
        )
    }

def compress_embeddings(embeddings_df, embed_name, min_components=10, max_components=100):
    """
    Compress the embeddings.
    
    Parameters:
    - embeddings_df: DataFrame containing the embeddings.
    - embed_name: Name of the embedding type for column naming.
    
    Returns:
    - compressed_embeddings: DataFrame with compressed embeddings.
    """
    
    print("Any infs:", np.isinf(embeddings_df.values).any())
    print("Any NaNs:", np.isnan(embeddings_df.values).any())

    embeddings_df = pd.DataFrame(embeddings_df, columns=embeddings_df.columns)

    pca = PCA()
    pca_embeddings = pca.fit_transform(embeddings_df.astype(np.float32))

    # filter based on explained variance
    threshold = 0.95
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance >= threshold) + 1
    n_components = min(max(n_components, min_components), max_components, len(explained_variance))

    print(f"PCA reduced to {n_components} components. (variance {explained_variance[n_components-1]:.2f})")

    # refit the PCA to only calculate the selected components
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings_df.astype(np.float32))

    output_df = pd.DataFrame(pca_embeddings, columns=[f'PCA_{embed_name}_{i}' for i in range(n_components)])

    # saving the PCA object for reuse on unseen data
    pkl.dump(pca, open(os.path.join(preproc_objs_dir, f'pca_{embed_name}_comp.pkl'), 'wb'))
    return output_df

def filter_embeddings(df, corr_threshold=0.9, verbose=False, embed_name=""):
    if 'target' in df.columns:
        df.drop(columns=['target'], inplace=True)
        
    original_shape = df.shape
    to_drop = []
    # Drop constant columns
    nunique = df.apply(pd.Series.nunique)
    constant_cols = nunique[nunique <= 1].index
    to_drop.extend(constant_cols)

    # df = df.drop(columns=constant_cols)

    # Drop columns with extreme magnitudes
    abs_max_vals = df.abs().max()
    unstable_cols = abs_max_vals[abs_max_vals > 1e6].index
    to_drop.extend(unstable_cols)

    # df = df.drop(columns=unstable_cols)

    # Remove columns with low variance
    low_variance_cols = df.var()[df.var() < 1e-6].index
    to_drop.extend(low_variance_cols)

    # df = df.drop(columns=low_variance_cols)

    # Drop highly correlated columns
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop.extend([col for col in upper.columns if any(upper[col] > corr_threshold)])
    to_drop = list(set(to_drop))  # unique

    # save the list of dropped columns for reference
    pkl.dump(to_drop, open(os.path.join(preproc_objs_dir, f'dropped_cols_{embed_name}.pkl'), 'wb'))
    with open(os.path.join(preproc_objs_dir, f'dropped_cols_{embed_name}.txt'), 'w') as f:
        for col in to_drop:
            f.write(f"{col}\n")
    
    df = df.drop(columns=to_drop)

    # Remove any remaining rows with inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if verbose:
        print(f"Original shape: {original_shape}")
        print(f"Removed constant columns: {len(constant_cols)}")
        print(f"Removed unstable magnitude columns: {len(unstable_cols)}")
        print(f"Removed highly correlated columns: {len(to_drop)}")
        print(f"Final shape: {df.shape}")
        print("-" * 40)

    return df
    
class KMeansFeatures(BaseEstimator, TransformerMixin):
    """
    Fit KMeans on numeric array (or dense-converted sparse).
    transform returns a 2-column numpy array: [cluster_id, dist_to_centroid]
    """
    def __init__(self, n_clusters=10, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        # ensure dense/numpy for kmeans fitting
        if hasattr(X, "iloc"):
            X_np = X.values
        elif sp.issparse(X):
            X_np = X.todense() if hasattr(X, "todense") else X.toarray()
        else:
            X_np = np.asarray(X)

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        self.kmeans_.fit(X_np)
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            X_np = X.values
        elif sp.issparse(X):
            X_np = X.todense() if hasattr(X, "todense") else X.toarray()
        else:
            X_np = np.asarray(X)

        labels = self.kmeans_.predict(X_np)
        distances = self.kmeans_.transform(X_np)
        dist_to_centroid = distances[np.arange(len(distances)), labels]
        out = np.vstack([labels, dist_to_centroid]).T
        return out  # numpy array with shape (n_samples, 2)

class KBinsDiscretizerDF(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, encode="ordinal", strategy="quantile", feature_names=None):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.feature_names = feature_names  # pass names of the input columns

    def fit(self, X, y=None):
        self.binner_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy, 
            quantile_method = 'linear'
        )
        self.binner_.fit(X)
        return self

    def transform(self, X):
        X_trans = self.binner_.transform(X)

        if sp.issparse(X_trans):
            X_trans = X_trans.toarray()

        n_output_bins = X_trans.shape[1]
        if self.feature_names is not None and self.encode.startswith("ordinal"):
            bin_names = []
            for i, fname in enumerate(self.feature_names):
                n_bins_feat = len(self.binner_.bin_edges_[i]) - 1
                for j in range(n_bins_feat):
                    bin_names.append(f"{fname}_bin{j}")

            bin_names = bin_names[:n_output_bins]        # in case of mismatch
        else:
            bin_names = [f"bin_{i}" for i in range(n_output_bins)]

        return pd.DataFrame(X_trans, columns=bin_names, index=getattr(X, "index", None))

class ToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.DataFrame(X)
    
def transform_dataset(
    dataset: pd.DataFrame
    ):

    df = dataset.copy().dropna()
    y = df.pop("target")
    X = df.drop(columns=["Canonical SMILES", "filename"], errors="ignore")

    pipeline, features = train_feature_pipeline(X, y, k=16, savepath="./processed_datasets/preproc_objs/feature_pipeline.pkl")
    X_train_transformed = pipeline.transform(X)

    # Fix: wrap output as DataFrame
    if isinstance(X_train_transformed, np.ndarray):
        # Try to get column names from transformers
        col_names = []

        # mostly fallbacks if transformers don't have get_feature_names_out
        X_selected = pipeline.named_steps['select_k'].transform(X)
        for name, trans in pipeline.named_steps['features'].transformer_list:
            if hasattr(trans, 'named_steps'):
                last_step = list(trans.named_steps.values())[-1]
                if hasattr(last_step, 'get_feature_names_out'):
                    col_names.extend(last_step.get_feature_names_out())
                else:
                    col_names.extend([f"{name}_{i}" for i in range(trans.transform(X_selected).shape[1])])
            else:
                col_names.extend([f"{name}_{i}" for i in range(trans.transform(X_selected).shape[1])])
        if len(col_names) != X_train_transformed.shape[1]:
            col_names = [f"feat_{i}" for i in range(X_train_transformed.shape[1])]
        X_train_transformed = pd.DataFrame(X_train_transformed, columns=col_names, index=X.index)

    # Add back target column
    X_train_transformed["target"] = y.values

    # concat to original data
    X_train_transformed = pd.concat([df.reset_index(drop=True), X_train_transformed.reset_index(drop=True)], axis=1)

    return X_train_transformed

def train_feature_pipeline(X, y, k=16, enable_binning=True, enable_clustering=True, savepath='feature_pipeline.pkl'):
    X = X.copy()
    X.columns = X.columns.astype(str)
    y = y.values if hasattr(y, "values") else y

    # Model-based selector: forest only for importances
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        max_features=k,
        threshold="mean"  
    )

    # Branches operate on the selected features only
    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("to_df", ToDataFrame())
    ])

    transformers = [("poly", poly)]

    if enable_binning:
        binner = Pipeline([
            ("binner", KBinsDiscretizerDF(n_bins=5, encode="ordinal", strategy="quantile")),
            ("to_df", ToDataFrame())
        ])
        transformers.append(("binning", binner))

    if enable_clustering:
        kmeans_pipe = Pipeline([
            ("kmeans", KMeansFeatures(n_clusters=10)),
            ("to_df", ToDataFrame())
        ])
        transformers.append(("kmeans", kmeans_pipe))

    pipeline = Pipeline([
        ("skew", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("select_k", selector),                      # RF used here only to select
        ("features", FeatureUnion(transformers)),    # build derived features from selected set
        ('scaler', StandardScaler()),                # scale all features
        ('to_df', ToDataFrame()),                    # ensure DataFrame output
    ])

    features = pipeline.fit_transform(X, y)
    joblib.dump({"pipeline": pipeline}, savepath)
    return pipeline, features

def generate_dataset_organic():
    embeddings = load_organic_embeddings()

    comp_c, comp_a, comp_all = [], [], []

    for name, (cation_df, anion_df) in embeddings.items():
        print(f"\nProcessing {name} embeddings...")

        cation_filtered = filter_embeddings(cation_df, embed_name=f"{name}_c", verbose=True)
        anion_filtered = filter_embeddings(anion_df, embed_name=f"{name}_a", verbose=True)

        compressed_c = compress_embeddings(cation_filtered, f"{name}_c")
        compressed_a = compress_embeddings(anion_filtered, f"{name}_a")

        comp_c.append(compressed_c)
        comp_a.append(compressed_a)
        comp_all.append(pd.concat([compressed_c, compressed_a], axis=1))

    # Concatenate all embeddings
    dataset_cations = pd.concat(comp_c, axis=1)
    dataset_anions = pd.concat(comp_a, axis=1)
    dataset_full = pd.concat(comp_all, axis=1)


    # raw_data = pd.read_csv(os.path.join(dataset_dir, "cytotoxicity_data.csv"))
    raw_data = pd.read_csv(os.path.join(dataset_dir, 'organic_ions.csv'))
    
    target = raw_data["target"]
    target = target.apply(lambda x: -np.log(x) if x > 0 else np.nan)

    # Add target 
    for df in [dataset_full, dataset_cations, dataset_anions]:
        df["target"] = target

    # add the respective ion smiles to the datasets. 
    dataset_anions['Canonical SMILES'] = raw_data['Anion']
    dataset_cations['Canonical SMILES'] = raw_data['Cation']

    dataset_anions_for_dedup = dataset_anions.copy()
    dataset_anions_for_dedup['Canonical SMILES'] = raw_data['Anion']
    dataset_anions_for_dedup['target'] = target
    
    dataset_cations_for_dedup = dataset_cations.copy()
    dataset_cations_for_dedup['Canonical SMILES'] = raw_data['Cation']
    dataset_cations_for_dedup['target'] = target
    
    dedup_cation_dataset = (
        dataset_cations_for_dedup.groupby('Canonical SMILES', as_index=False)
        .agg(lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    )

    dedup_anion_dataset = (
        dataset_anions_for_dedup.groupby('Canonical SMILES', as_index=False)
        .agg(lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    )
    print("shapes after deduplication:")
    print("Cations:", dedup_cation_dataset.shape)
    print("Anions:", dedup_anion_dataset.shape)
    print("Full:", dataset_full.shape)
    dataset_full = transform_dataset(dataset_full)
    dataset_cations = transform_dataset(dedup_cation_dataset)
    dataset_anions = transform_dataset(dedup_anion_dataset)

    print("\n Final dataset shapes:")
    print("Full:", dataset_full.shape)
    print("Cations:", dataset_cations.shape)
    print("Anions:", dataset_anions.shape)

    dataset_full.to_csv(os.path.join(dataset_dir, 'final_dataset_org.csv'), index=False)
    dataset_cations.to_csv(os.path.join(dataset_dir, "final_dataset_cations_org.csv"), index=False)
    dataset_anions.to_csv(os.path.join(dataset_dir, "final_dataset_anions_org.csv"), index=False)

def generate_dataset_all():
    embeddings = load_embeddings()

    comp_c, comp_a, comp_all = [], [], []

    for name, (cation_df, anion_df) in embeddings.items():
        print(f"\nProcessing {name} embeddings...")

        cation_filtered = filter_embeddings(cation_df, embed_name=f"{name}_c")
        anion_filtered = filter_embeddings(anion_df, embed_name=f"{name}_a")

        compressed_c = compress_embeddings(cation_filtered, f"{name}_c")
        compressed_a = compress_embeddings(anion_filtered, f"{name}_a")

        comp_c.append(compressed_c)
        comp_a.append(compressed_a)
        comp_all.append(pd.concat([compressed_c, compressed_a], axis=1))

    # Concatenate all embeddings
    dataset_cations = pd.concat(comp_c, axis=1).add_prefix("cation_")
    dataset_anions = pd.concat(comp_a, axis=1).add_prefix("anion_")
    dataset_full = pd.concat(comp_all, axis=1)

    # Scale the final datasets
    scaler_full = StandardScaler()
    dataset_full = pd.DataFrame(scaler_full.fit_transform(dataset_full), columns=dataset_full.columns)

    scaler_cation = StandardScaler()
    dataset_cations = pd.DataFrame(scaler_cation.fit_transform(dataset_cations), columns=dataset_cations.columns)

    scaler_anion = StandardScaler()
    dataset_anions = pd.DataFrame(scaler_anion.fit_transform(dataset_anions), columns=dataset_anions.columns)

    # save the scaler objects for future processing
    pkl.dump(scaler_full, open(os.path.join(preproc_objs_dir, 'scaler_full.pkl'), 'wb'))
    pkl.dump(scaler_cation, open(os.path.join(preproc_objs_dir, 'scaler_cation.pkl'), 'wb'))
    pkl.dump(scaler_anion, open(os.path.join(preproc_objs_dir, 'scaler_anion.pkl'), 'wb'))

    # Add target + meta-data
    cytotoxicity_data = pd.read_csv(os.path.join(dataset_dir, "cytotoxicity_data.csv"))
    for df in [dataset_full, dataset_cations, dataset_anions]:
        target = cytotoxicity_data["CC50/IC50/EC50, mM"]
        target = target.apply(lambda x: -np.log(x) if x > 0 else np.nan)

        df["target"] = target
        df["Canonical SMILES"] = cytotoxicity_data["Canonical SMILES"]
        df["filename"] = cytotoxicity_data["filename"]


    dataset_full = transform_dataset(dataset_full)
    dataset_cations = transform_dataset(dataset_cations)
    dataset_anions = transform_dataset(dataset_anions)

    print(type(dataset_full))
    # Save datasets
    dataset_full.to_csv(os.path.join(dataset_dir, "final_dataset.csv"), index=False)
    dataset_cations.to_csv(os.path.join(dataset_dir, "final_dataset_cations.csv"), index=False)
    dataset_anions.to_csv(os.path.join(dataset_dir, "final_dataset_anions.csv"), index=False)

    print("\n Final dataset shapes:")
    print("Full:", dataset_full.shape)
    print("Cations:", dataset_cations.shape)
    print("Anions:", dataset_anions.shape)


def main():
    generate_dataset_organic()

if __name__ == "__main__":
    main()
    print("Datasets generated and saved to processed_datasets")