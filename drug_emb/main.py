from typing import Dict

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from data_module import sparse_molecular_data_module
from module import molgan


def load_configuration():
    # Load configuration file(.yaml)
    with open("data/config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Set parameters
    project_params = cfg["project_params"]
    module_params = cfg["module_params"]
    data_module_params = cfg["data_module_params"]

    return project_params, module_params, data_module_params


def load_net_and_data(module_params: Dict, data_module_params: Dict):
    # Set model and dataloader
    data = sparse_molecular_data_module(**data_module_params)
    # Update model configuration
    update_dict = {
        "atom_dim": data.dataset.atom_num_types,
        "bond_dim": data.dataset.bond_num_types,
        "vertex": data.dataset.vertex,
        "data_path": data_module_params["dataset_params"]["data_dir"]
        + data_module_params["dataset_params"]["data_name"].split(".")[0]
        + "_preprocess.npz",
    }
    module_params.update(update_dict)
    net = molgan(**module_params)
    return net, data


def load_trainer(project_params) -> pl.Trainer:
    # Set wandb
    pl.seed_everything(project_params["seed"])
    trainer = pl.Trainer(
        logger=WandbLogger(
            project=project_params["project"],
            entity=project_params["entity"],
            log_model=True,
        ),
        gpus=project_params["gpus"],
        accelerator=project_params["accelerator"],
        max_epochs=project_params["max_epochs"],
        callbacks=[
            EarlyStopping(
                patience=project_params["patience"], monitor=project_params["monitor"]
            ),
            LearningRateMonitor(logging_interval=project_params["logging_interval"]),
            ModelCheckpoint(
                dirpath=project_params["save_path"],
                filename="MolGAN",
                monitor=project_params["monitor"],
                save_top_k=1,
                mode="min",
            ),
        ],
    )
    return trainer


def main():
    project_params, module_params, dataset_params = load_configuration()

    net, data = load_net_and_data(module_params, dataset_params)

    trainer = load_trainer(project_params)
    trainer.fit(net, data)


if __name__ == "__main__":
    main()
