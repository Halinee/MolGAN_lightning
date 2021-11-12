import gzip
import math
import pickle

import numpy as np
from PIL import Image as pilimg
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Crippen, Draw
import torch as th
import torch.nn.functional as F


def gradient_penalty(x, y, device):
    """Compute gradient penalty: (L2_norm(dy / dx) - 1) ** 2"""
    weight = th.ones(y.size()).to(device)
    dy_dx = th.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=weight,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]

    dy_dx = dy_dx.view(dy_dx.size(0), -1)
    dy_dx_l2_norm = th.sqrt(th.sum(dy_dx ** 2, dim=1))
    return th.mean((dy_dx_l2_norm - 1) ** 2)


def onehot_encoding(labels, dim, device):
    """Convert label indices to one-hot vectors."""
    out = th.zeros(list(labels.size()) + [dim]).to(device)
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.0)
    return out


def postprocess(inputs, method, temperature=1.0):
    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    if method == "soft_gumbel":
        softmax = [
            F.gumbel_softmax(
                e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                hard=False,
            ).view(e_logits.size())
            for e_logits in listify(inputs)
        ]
    elif method == "hard_gumbel":
        softmax = [
            F.gumbel_softmax(
                e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                hard=True,
            ).view(e_logits.size())
            for e_logits in listify(inputs)
        ]
    else:
        softmax = [
            F.softmax(e_logits / temperature, -1) for e_logits in listify(inputs)
        ]

    return [delistify(e) for e in (softmax)]


def matrices_to_mol(self, node_labels, edge_labels, strict=False):
    mol = Chem.RWMol()

    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end:
            mol.AddBond(
                int(start), int(end), self.bond_decoder_m[edge_labels[start, end]]
            )

    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None

    return mol


def generate_mols_img(mols, sub_img_size=(512, 512), legends=None, row=2, **kwargs):
    if legends is None:
        legends = [None] * len(mols)
    res = pilimg.new(
        "RGBA",
        (
            sub_img_size[0] * row,
            sub_img_size[1] * (len(mols) // row)
            if len(mols) % row == 0
            else sub_img_size[1] * ((len(mols) // row) + 1),
        ),
    )
    for i, mol in enumerate(mols):
        res.paste(
            Draw.MolToImage(mol, sub_img_size, legend=legends[i], **kwargs),
            ((i // row) * sub_img_size[0], (i % row) * sub_img_size[1]),
        )

    return res


def reward(mols, data_smiles, metric):
    rr = 1.0
    for m in ("logp,sas,qed,unique" if metric == "all" else metric).split(","):

        if m == "np":
            rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
        elif m == "logp":
            rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(
                mols, norm=True
            )
        elif m == "sas":
            rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
        elif m == "qed":
            rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(
                mols, norm=True
            )
        elif m == "novelty":
            rr *= MolecularMetrics.novel_scores(mols, data_smiles)
        elif m == "dc":
            rr *= MolecularMetrics.drugcandidate_scores(mols, data_smiles)
        elif m == "unique":
            rr *= MolecularMetrics.unique_scores(mols)
        elif m == "diversity":
            rr *= MolecularMetrics.diversity_scores(mols, data_smiles)
        elif m == "validity":
            rr *= MolecularMetrics.valid_scores(mols)
        else:
            raise RuntimeError("{} is not defined as a metric".format(m))

    return rr.reshape(-1, 1)


class MolecularMetrics(object):
    @staticmethod
    def np_model(model_dir):
        return pickle.load(gzip.open(model_dir))

    @staticmethod
    def sa_model(model_dir):
        return {
            i[j]: float(i[0])
            for i in pickle.load(gzip.open(model_dir))
            for j in range(1, len(i))
        }

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ""

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ""
        return x is not None and "*" not in s and "." not in s and s != ""

    @staticmethod
    def valid_scores(mols):
        return np.array(
            list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32
        )

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(
            list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32
        ).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(
                map(
                    lambda x: MolecularMetrics.valid_lambda(x)
                    and Chem.MolToSmiles(x) not in data.smiles,
                    mols,
                )
            )
        )

    @staticmethod
    def novel_filter(mols, data):
        return list(
            filter(
                lambda x: MolecularMetrics.valid_lambda(x)
                and Chem.MolToSmiles(x) not in data.smiles,
                mols,
            )
        )

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(
            MolecularMetrics.valid_filter(mols), data
        ).mean()

    @staticmethod
    def unique_scores(mols):
        smiles = list(
            map(
                lambda x: Chem.MolToSmiles(x)
                if MolecularMetrics.valid_lambda(x)
                else "",
                mols,
            )
        )
        return np.clip(
            0.75
            + np.array(
                list(map(lambda x: 1 / smiles.count(x) if x != "" else 0, smiles)),
                dtype=np.float32,
            ),
            0,
            1,
        )

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    @staticmethod
    def natural_product_scores(mols, norm=False):

        # calculating the score
        scores = [
            sum(
                MolecularMetrics.np_model("data/NP_score.pkl.gz").get(bit, 0)
                for bit in Chem.rdMolDescriptors.GetMorganFingerprint(
                    mol, 2
                ).GetNonzeroElements()
            )
            / float(mol.GetNumAtoms())
            if mol is not None
            else None
            for mol in mols
        ]

        # preventing score explosion for exotic molecules
        scores = list(
            map(
                lambda score: score
                if score is None
                else (
                    4 + math.log10(score - 4 + 1)
                    if score > 4
                    else (-4 - math.log10(-4 - score + 1) if score < -4 else score)
                ),
                scores,
            )
        )

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = (
            np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores
        )

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(
            list(
                map(
                    lambda x: 0 if x is None else x,
                    [
                        MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol))
                        if mol is not None
                        else None
                        for mol in mols
                    ],
                )
            )
        )

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [
            MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol))
            if mol is not None
            else None
            for mol in mols
        ]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = (
            np.clip(
                MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0
            )
            if norm
            else scores
        )

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += MolecularMetrics.sa_model("data/SA_score.pkl.gz").get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0

        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = (
            0.0
            - sizePenalty
            - stereoPenalty
            - spiroPenalty
            - bridgePenalty
            - macrocyclePenalty
        )

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.0
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * 0.5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
        # smooth the 10-end
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [
            MolecularMetrics._compute_SAS(mol) if mol is not None else None
            for mol in mols
        ]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = (
            np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0)
            if norm
            else scores
        )

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        fps = [
            Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            for mol in rand_mols
        ]

        scores = np.array(
            list(
                map(
                    lambda x: MolecularMetrics.__compute_diversity(x, fps)
                    if x is not None
                    else 0,
                    mols,
                )
            )
        )
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 4, nBits=2048
        )
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):

        scores = (
            MolecularMetrics.constant_bump(
                MolecularMetrics.water_octanol_partition_coefficient_scores(
                    mols, norm=True
                ),
                0.210,
                0.945,
            )
            + MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            + MolecularMetrics.novel_scores(mols, data)
            + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3
        ) / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(
            condlist=[x <= x_low, x >= x_high],
            choicelist=[
                np.exp(-((x - x_low) ** 2) / decay),
                np.exp(-((x - x_high) ** 2) / decay),
            ],
            default=np.ones_like(x),
        )
