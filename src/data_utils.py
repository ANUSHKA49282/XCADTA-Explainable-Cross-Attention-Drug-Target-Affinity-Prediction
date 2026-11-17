
import os
import hashlib
import pickle
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from rdkit import Chem


# Atom features (65 features for each atom)

ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
]

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    'other'
]


# Convert a value into a one-hot encoded list.
def _one_hot(value, choices) -> List[int]:
    if value not in choices:
        value = choices[-1]
    return [1 if value == c else 0 for c in choices]


# Get the 65 features of an atom.
def atom_features(atom) -> np.ndarray:
    symbol = atom.GetSymbol()

    features = (
        _one_hot(symbol, ATOM_SYMBOLS)
        + _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetFormalCharge()]
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [1 if atom.GetIsAromatic() else 0]
        + _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + [1 if atom.HasProp('_ChiralityPossible') else 0]
    )

    return np.array(features, dtype=np.float32)


# Convert a SMILES string into a molecular graph.
def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

    atoms = mol.GetAtoms()
    node_feats = np.stack([atom_features(a) for a in atoms], axis=0)

    edges = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    if len(edges) == 0:
        edges = [(0, 0)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)


# Generate a hash for a protein sequence.
def sequence_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode('utf-8')).hexdigest()[:16]


# Load the KIBA dataset files.
def load_kiba_raw(data_dir: str):
    import json

    with open(os.path.join(data_dir, 'ligands_can.txt')) as f:
        ligands = json.load(f)

    with open(os.path.join(data_dir, 'proteins.txt')) as f:
        proteins = json.load(f)

    with open(os.path.join(data_dir, 'Y'), 'rb') as f:
        Y = pickle.load(f, encoding='latin1')

    return ligands, proteins, Y


# Create drug-protein pairs with their affinity values.
def build_pair_list(ligands: Dict, proteins: Dict, Y) -> List[Tuple[str, str, float]]:
    drug_ids = list(ligands.keys())
    protein_ids = list(proteins.keys())

    pairs = []

    for di, drug_id in enumerate(drug_ids):
        for pi, protein_id in enumerate(protein_ids):
            val = Y[di][pi]

            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                pairs.append((ligands[drug_id], proteins[protein_id], float(val)))

    return pairs


# Select pairs while maintaining protein diversity.
def stratified_subsample(
    pairs: List[Tuple[str, str, float]],
    target_size: int = 20000,
    seed: int = 42
) -> List[Tuple[str, str, float]]:

    random.seed(seed)

    by_protein: Dict[str, list] = {}

    for smiles, seq, y in pairs:
        by_protein.setdefault(seq, []).append((smiles, seq, y))

    proteins = list(by_protein.keys())
    random.shuffle(proteins)

    n_proteins = len(proteins)
    per_protein_quota = max(1, target_size // n_proteins)

    sampled = []

    for seq in proteins:
        bucket = by_protein[seq]
        random.shuffle(bucket)
        sampled.extend(bucket[:per_protein_quota])

    random.shuffle(sampled)

    if len(sampled) > target_size:
        sampled = sampled[:target_size]

    return sampled


# Split the data into training, validation and test sets.
def stratified_split(
    pairs: List[Tuple[str, str, float]],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
):

    random.seed(seed)

    by_protein: Dict[str, list] = {}

    for p in pairs:
        by_protein.setdefault(p[1], []).append(p)

    train, val, test = [], [], []

    for seq, bucket in by_protein.items():
        random.shuffle(bucket)

        n = len(bucket)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# Load molecular graphs and protein embeddings.
class XCADTADataset(Dataset):

    def __init__(
        self,
        pairs: List[Tuple[str, str, float]],
        esm_embeddings: Dict[str, torch.Tensor],
        l_max: int = 1000
    ):
        self.pairs = pairs
        self.esm_embeddings = esm_embeddings
        self.l_max = l_max
        self._graph_cache: Dict[str, Data] = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        smiles, seq, affinity = self.pairs[idx]

        if smiles not in self._graph_cache:
            self._graph_cache[smiles] = smiles_to_graph(smiles)

        graph = self._graph_cache[smiles]

        h = sequence_hash(seq)

        if h not in self.esm_embeddings:
            raise KeyError(
                f"No cached ESM-2 embedding for sequence hash {h}. "
                f"Run esm_embed.py first to precompute and cache embeddings."
            )

        protein_emb = self.esm_embeddings[h][:self.l_max]

        return {
            'graph': graph,
            'protein_emb': protein_emb,
            'affinity': torch.tensor(affinity, dtype=torch.float),
        }


# Prepare graphs and protein embeddings for each batch.
def collate_fn(batch):
    from torch_geometric.data import Batch

    graphs = Batch.from_data_list([b['graph'] for b in batch])
    affinities = torch.stack([b['affinity'] for b in batch])

    max_len = max(b['protein_emb'].shape[0] for b in batch)
    esm_dim = batch[0]['protein_emb'].shape[1]

    padded = torch.zeros(len(batch), max_len, esm_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b['protein_emb'].shape[0]
        padded[i, :L] = b['protein_emb']
        mask[i, :L] = True

    return {
        'graphs': graphs,
        'protein_emb': padded,
        'protein_mask': mask,
        'affinity': affinities,
    }
