"""
Author: Muhammad Faizan

---------------
python train.py
"""
# ====================================================================
import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf.omegaconf import OmegaConf
from dataset import Dataset
from model import colaModel
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class visualizationLogger(pl.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    # on each validation phase logs predictions results to wandb
    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.dataset.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        logits = outputs.logits
        predictions = torch.argmax(logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"sentence": sentences, "Label": labels.numpy(), "Predictions": predictions.numpy()
            })
        wrong_df = df[df["Label"] != df["Predictions"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

# run trainer.
@hydra.main(config_path = "./configs", config_name = "configs", version_base = None)
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve = True))
    logger.info(f"Using model: {cfg.model.name}")
    logger.info(f"using the tokenizer: {cfg.model.tokenizer}")

    # instantiate two instance like the dataset and model
    cola_dataset = Dataset(cfg.model.tokenizer, cfg.preprocess.batch, 
                           cfg.preprocess.max_length)
    cola_model = colaModel(cfg.model.name)

    root_dir = hydra.utils.get_original_cwd()

    # add model checkpoint and model early stoppoing callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath = f"{root_dir}/models", monitor = "valid/loss", mode = "min",
        filename = "best-checkpoint"
    )
    
     #BUG: fixed 
    early_stoppoing_callback = EarlyStopping(
        monitor = "valid/loss", patience = 3, verbose = True, mode = "min"
    )

    # adding wandb logger
    wandb_logger = WandbLogger(project = "MLOps - logging and experiment tracking",
                               entity = "engrfaizan-ai")
    

    # now create a trainer object for training
    trainer = pl.Trainer(
        logger = wandb_logger,
        max_epochs = cfg.training.max_epochs,
        log_every_n_steps = cfg.training.log_every_n_steps,
        deterministic = cfg.training.deterministic,
        callbacks = [checkpoint_callback, visualizationLogger(cola_dataset), early_stoppoing_callback],
        # limit_train_batches = cfg.training.limit_train_batches,
        # limit_val_batches = cfg.training.limit_val_batches,
        ) 

    # fit on the dataset
    trainer.fit(cola_model, datamodule=cola_dataset)
    wandb.finish()

#TODO: TO TRAIN THE MODEL
if __name__ == "__main__":
    main()
