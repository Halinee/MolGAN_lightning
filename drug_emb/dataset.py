import os.path as osp

import numpy as np
from rdkit import Chem
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm


class sparse_molecular_dataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        data_name: str = None,
        add_h: bool = False,
        filters: int = 30,
    ):
        data = self.load(data_dir, data_name, add_h, filters)
        atom_labels, bond_labels, smiles_labels = self.generate_labels(data)
        (
            self.atom_encoder,
            self.atom_decoder,
            self.atom_num_types,
        ) = self.generate_encoder_decoder(atom_labels, "atom")
        (
            self.bond_encoder,
            self.bond_decoder,
            self.bond_num_types,
        ) = self.generate_encoder_decoder(bond_labels, "bond")
        (
            self.smiles_encoder,
            self.smiles_decoder,
            self.smiles_num_types,
        ) = self.generate_encoder_decoder(smiles_labels, "smiles")

        if osp.exists(data_dir + data_name.split(".")[0] + "_preprocess_f" + str(filters) + ".npz"):
            data_adj = np.load(
                data_dir + data_name.split(".")[0] + "_preprocess_f" + str(filters) + ".npz",
                allow_pickle=True,
            )
            data, data_smiles, data_A, data_X, data_F = (
                data_adj["mols"],
                data_adj["smiles"],
                data_adj["data_A"],
                data_adj["data_X"],
                data_adj["data_F"],
            )
        else:
            data, data_smiles, data_A, data_X, data_F = self.generate_adj_matrix(data)
            np.savez(
                data_dir + data_name.split(".")[0] + "_preprocess.npz",
                mols=np.array(data),
                smiles=np.array(data_smiles),
                data_A=np.array(data_A),
                data_X=np.array(data_X),
                data_F=np.array(data_F),
            )

        self.data_idx = np.arange(len(data))
        self.data_smiles = np.array(data_smiles)
        self.data_A = np.stack(data_A)
        self.data_X = np.stack(data_X)
        self.data_F = np.stack(data_F)
        self.vertex = self.data_F.shape[-2]

    def __getitem__(self, idx):
        # Raw data
        mol_idx = self.data_idx[idx]
        A = self.data_A[idx]
        X = self.data_X[idx]
        # Preprocessed data (numpy to tensor)
        A = th.as_tensor(A, dtype=th.long).unsqueeze(0)
        X = th.as_tensor(X, dtype=th.long).unsqueeze(0)
        return mol_idx, A, X

    def __len__(self):
        return len(self.data_idx)

    def load(self, data_dir, data_name, add_h, filters):
        print("Loading data...")
        if osp.exists(data_dir + data_name.split(".")[0] + "_" + str(filters) + ".npy"):
            origin_data = np.load(
                data_dir + data_name.split(".")[0] + "_" + str(filters) + ".npy", allow_pickle=True
            ).tolist()
        else:
            if data_name.endswith(".sdf"):
                origin_data = list(
                    filter(
                        lambda x: x is not None,
                        Chem.SDMolSupplier(data_dir + data_name),
                    )
                )
            elif data_name.endswith(".smi"):
                origin_data = [
                    Chem.MolFromSmiles(line)
                    for line in open(data_dir + data_name, "r").readlines()
                ]
            else:
                raise ValueError("Input file format must be .sdf or .smi")

            np.save(
                data_dir + data_name.split(".")[0] + "_" + str(filters) + ".npy",
                np.array(origin_data),
                allow_pickle=True,
            )

        data = list(map(Chem.AddHs, origin_data)) if add_h else origin_data
        data = list(filter(lambda x: x.GetNumAtoms() <= filters, data))
        print(
            "Extracted {} out of {} molecules {}adding Hydrogen!\n".format(
                len(data), len(origin_data), "" if add_h else "not "
            )
        )

        return data

    def generate_labels(self, data):
        atom_labels = sorted(
            set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0])
        )
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(
            sorted(set(bond.GetBondType() for mol in data for bond in mol.GetBonds()))
        )
        smiles_labels = ["E"] + list(
            set(c for mol in data for c in Chem.MolToSmiles(mol))
        )

        return atom_labels, bond_labels, smiles_labels

    def generate_encoder_decoder(self, labels, type):
        print("Creating {} encoder...".format(type))
        encoder = {l: i for i, l in tqdm(enumerate(labels))}
        print("Created {} encoder!\n\nCreating {} decoder...".format(type, type))
        decoder = {i: l for i, l in tqdm(enumerate(labels))}
        print("Created {} decoder!\n".format(type))
        num_types = len(labels)

        return encoder, decoder, num_types

    def generate_adj_matrix(self, origin_data):
        data = []
        data_smiles = []
        data_A = []
        data_X = []
        data_F = []
        max_length = max(mol.GetNumAtoms() for mol in origin_data)

        print("Creating adjacency matrices...")
        for i, mol in tqdm(enumerate(origin_data)):
            A = self.generate_A(mol, connected=True, max_length=max_length)
            if A is not None:
                data.append(mol)
                data_smiles.append(Chem.MolToSmiles(mol))
                data_A.append(A)
                data_X.append(self.generate_X(mol, max_length=max_length))
                data_F.append(self.generate_F(mol, max_length=max_length))

        print(
            "Created {} features and adjacency matrices  out of {} molecules!\n".format(
                len(data), len(origin_data)
            )
        )

        return data, data_smiles, data_A, data_X, data_F

    def generate_A(self, mol, connected=True, max_length=None):
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [
            b.GetEndAtomIdx() for b in mol.GetBonds()
        ]
        bond_type = [self.bond_encoder[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[: mol.GetNumAtoms(), : mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def generate_X(self, mol, max_length=None):
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array(
            [self.atom_encoder[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
            + [0] * (max_length - mol.GetNumAtoms()),
            dtype=np.int32,
        )

    def generate_F(self, mol, max_length=None):
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        features = np.array(
            [
                [
                    *[a.GetDegree() == i for i in range(5)],
                    *[a.GetExplicitValence() == i for i in range(9)],
                    *[int(a.GetHybridization()) == i for i in range(1, 7)],
                    *[a.GetImplicitValence() == i for i in range(9)],
                    a.GetIsAromatic(),
                    a.GetNoImplicit(),
                    *[a.GetNumExplicitHs() == i for i in range(5)],
                    *[a.GetNumImplicitHs() == i for i in range(5)],
                    *[a.GetNumRadicalElectrons() == i for i in range(5)],
                    a.IsInRing(),
                    *[a.IsInRingSize(i) for i in range(2, 9)],
                ]
                for a in mol.GetAtoms()
            ],
            dtype=np.int32,
        )

        return np.vstack(
            (features, np.zeros((max_length - features.shape[0], features.shape[1])))
        )
