"""
Convert the model to the ONNX format
-----------------------------------

Author: Muahmmad Faizan
----------------------
python convert2onnx.py -h
"""
import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf
from model import colaModel
from dataset import Dataset

# configure logger
logger = logging.getLogger(__name__)

# convert to onnx foramt
@hydra.main(config_path="configs/", config_name='configs.yaml', version_base= None)
def convert(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"loading pretrained model from {model_path}")
    model = colaModel.load_from_checkpoint(model_path)

    data = Dataset(model=cfg.model.tokenizer, batch_size= cfg.preprocess.batch,
                   max_length= cfg.preprocess.max_length)
    
    data.setup()
    input_batch = next(iter(data.train_dataloader()))
    input_example = {"input_ids": input_batch["input_ids"][0].unsqueeze(0),
                     "attention_mask": input_batch["attention_mask"][0].unsqueeze(0)}
    # Now Export the model to onnx format
    logger.info('exporting the model to onnx.')
    torch.onnx.export(
        model, # model to be exported.
    (
        input_example["input_ids"],
        input_example["attention_mask"]
    ), # input to the model
     f"{root_dir}/models/model.onnx",
     export_params= True,
     opset_version= 10, 
     input_names= ["input_ids", "attention_mask"],
     output_names= ["output"],
     dynamic_axes= {
         "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
     },
    )
    logger.info(f'The model has been sucessfully converted. location: {root_dir}/models/model.onnx')

if __name__ == '__main__':
    convert()
