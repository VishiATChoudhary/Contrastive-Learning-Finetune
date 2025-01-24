import pytorch_lightning as pl
import torch
import yaml
import argparse
import sys
sys.path.append(r"C:\Users\Vishi\VSC Codes\CheXzero Finetune\VLP-Seminar\Finetune")
from methods.cls_model import FinetuneClassifier
from datasets.cls_dataset import ChexPertImageClsDataset
from datasets.data_module import DataModule
from datasets.transforms import DataTransforms
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

chZ_CP_ft_ckpt = r'C:\Users\Vishi\VSC Codes\CheXzero Finetune\VLP-Seminar\data\ckpts\FinetuneCLS\chexpert\2025_01_19_18_37_48\epoch=0-step=3870.ckpt'
vanilla_chZ = r'C:\Users\Vishi\VSC Codes\CheXzero Finetune\VLP-Seminar\checkpoints\cheXzero\visual.ckpt'
vanilla_vit = r'C:\Users\Vishi\VSC Codes\CheXzero Finetune\VLP-Seminar\checkpoints\vit\vit_base.ckpt'
vanilla_resnet = r'C:\Users\Vishi\VSC Codes\CheXzero Finetune\VLP-Seminar\checkpoints\resnet\resnet50.ckpt'

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lightning Inference")
    parser.add_argument("--dataset", type=str, default="chexpert", help="Dataset to use: rsna, chexpert")
    parser.add_argument('--config', type=str, default='configs/chexpert.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=vanilla_chZ, help='Path to trained model checkpoint') # change checkpoint path here
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Load dataset
    if args.dataset == "chexpert":
        datamodule = DataModule(dataset=ChexPertImageClsDataset, 
                                config=config, collate_fn=None,
                                transforms=DataTransforms, 
                                data_pct=1.0,  # Use full dataset for inference
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers)
    else:
        raise ValueError("Unsupported dataset. Use 'chexpert'.")

    # Load pre-trained model
    model = FinetuneClassifier(config)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict'], strict=False) use with finetuned ckpts
    try:
        model.load_state_dict(checkpoint['model'], strict=False) # use with non finetuned ckpts
    except:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()  # Set model to evaluation mode

    # Run inference
    trainer = pl.Trainer(accelerator='cpu')
    # import pdb; pdb.set_trace()
    trainer.test(model, datamodule.test_dataloader())


