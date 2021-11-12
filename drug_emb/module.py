from typing import List

import numpy as np
import pytorch_lightning as pl
from torch import optim
from wandb import Image

from blocks import generator, discriminator
from utility import *


class molgan(pl.LightningModule):
    def __init__(
        self,
        z_dim: int = 8,
        atom_dim: int = None,
        bond_dim: int = None,
        vertex: int = None,
        g_conv_dim: List = [128, 256, 512],
        d_conv_dim: List = [[128, 64], 128, [128, 64]],
        lambda_gp: int = 10,
        post_method: str = "softmax",
        dropout: float = 0.1,
        g_lr: float = 0.0001,
        d_lr: float = 0.0001,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        num_iter_decay: int = 100,
        gamma: float = 0.5,
        data_path: str = None,
    ):
        """Initialize configurations."""
        super(molgan, self).__init__()
        self.save_hyperparameters()
        # Specific data configurations
        preprocess_data = np.load(data_path, allow_pickle=True)
        self.mol_data = preprocess_data["mols"]
        self.smiles_data = preprocess_data["smiles"]
        # Validation configurations
        self.validation_z = th.randn(8, z_dim)
        # Build model
        self.generator = generator(
            g_conv_dim,
            z_dim,
            vertex,
            atom_dim,
            bond_dim,
            dropout,
        )
        self.discriminator = discriminator(d_conv_dim, atom_dim, bond_dim, dropout)
        self.rewarder = discriminator(d_conv_dim, atom_dim, bond_dim, dropout)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Preprocess input data
        mol_idx, A, X = batch
        mols = self.mol_data[mol_idx.cpu()]
        smiles = self.smiles_data[mol_idx.cpu()]
        adjacency = onehot_encoding(A, self.hparams.bond_dim, self.device)
        features = onehot_encoding(X, self.hparams.atom_dim, self.device)
        z = (
            th.randn(mol_idx.shape[0], self.hparams.z_dim)
            .to(th.float32)
            .to(self.device)
        )
        # Train generator
        if optimizer_idx == 0:
            # Z to target
            edges, nodes = self(z)
            # Postprocess with Gumbel softmax
            edges_hat, nodes_hat = postprocess((edges, nodes), self.hparams.post_method)
            print("input layer shape")
            print(self.hparams.d_conv_dim[0][-1] + self.hparams.bond_dim)
            logits_fake, features_fake = self.discriminator(edges_hat, None, nodes_hat)
            g_loss_fake = -th.mean(logits_fake)
            # Real reward
            real_reward = th.from_numpy(reward(mols, smiles, metric="all")).to(
                self.device
            )
            # Fake reward
            edges_hard, nodes_hard = postprocess((edges, nodes), method="hard_gumbel")
            edges_hard, nodes_hard = (
                th.max(edges_hard, -1)[1],
                th.max(nodes_hard, -1)[1],
            )
            mols = [
                matrices_to_mol(
                    n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True
                )
                for e_, n_ in zip(edges_hard, nodes_hard)
            ]
            fake_reward = th.from_numpy(reward(mols, smiles, metric="all")).to(
                self.device
            )
            # Value loss
            real_value_logits, _ = self.rewarder(adjacency, None, features, th.sigmoid)
            fake_value_logits, _ = self.rewarder(edges_hat, None, nodes_hat, th.sigmoid)
            g_loss_value = th.mean(
                (real_value_logits - real_reward) ** 2
                + (fake_value_logits - fake_reward) ** 2
            )
            # Generator loss
            g_loss = g_loss_fake + g_loss_value
            self.log(
                {
                    "g_loss": g_loss,
                    "g_loss_fake": g_loss_fake,
                    "g_loss_value": g_loss_value,
                }
            )
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            # Compute loss with real data
            logits_real, features_real = self.discriminator(adjacency, None, features)
            d_loss_real = -th.mean(logits_real)
            # Compute loss with fake data
            edges, nodes = self(z)
            # Postprocess with Gumbel softmax
            edges_hat, nodes_hat = postprocess((edges, nodes), self.hparams.post_method)
            logits_fake, features_fake = self.discriminator(edges_hat, None, nodes_hat)
            d_loss_fake = -th.mean(logits_fake)
            # Compute loss for gradient penalty
            eps = th.rand(logits_real.size(0), 1, 1, 1).to(self.device)
            x_int_0 = (eps * adjacency + (1.0 - eps) * edges_hat).requires_grad_(True)
            x_int_1 = (
                eps.squeeze(-1) * features + (1.0 - eps.squeeze(-1)) * nodes_hat
            ).requires_grad_(True)
            grad_0, grad_1 = self.discriminator(x_int_0, None, x_int_1)
            d_loss_gp = gradient_penalty(grad_0, x_int_0) + gradient_penalty(
                grad_1, x_int_1
            )
            # Discriminator loss
            d_loss = d_loss_fake + d_loss_real + self.hparams.lambda_gp * d_loss_gp
            self.log(
                {
                    "d_loss": d_loss,
                    "d_loss_real": d_loss_real,
                    "d_loss_fake": d_loss_fake,
                }
            )
            return d_loss

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.layers[0].weight).to(self.device)
        # Log sampled molecule
        sample_edges, sample_nodes = self(z)
        sample_mols = [
            matrices_to_mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
            for e_, n_ in zip(sample_edges, sample_nodes)
        ]
        self.logger.experiment.log(
            {"Generated Molecules": [Image(generate_mols_img(sample_mols))]}
        )

    def configure_optimizers(self):
        g_opt = optim.Adam(
            list(self.generator.parameters()) + list(self.rewarder.parameters()),
            self.hparams.g_lr,
            [self.hparams.beta_1, self.hparams.beta_2],
        )
        d_opt = optim.Adam(
            self.discriminator.parameters(),
            self.hparams.d_lr,
            [self.hparams.beta_1, self.hparams.beta_2],
        )
        g_sch = optim.lr_scheduler.StepLR(
            g_opt, step_size=self.hparams.num_iter_decay, gamma=0.5
        )
        d_sch = optim.lr_scheduler.StepLR(
            d_opt, step_size=self.hparams.num_iter_decay, gamma=0.5
        )

        return [g_opt, d_opt], [g_sch, d_sch]
