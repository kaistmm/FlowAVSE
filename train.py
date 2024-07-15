import argparse
from argparse import ArgumentParser
import os
import sys
import glob

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module_vi import SpecsDataModule
from sgmse.model import StochasticRegenerationModel

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

import numpy as np
import random

class CheckpointEveryNSteps(pl.Callback):
    """
	from https://github.com/Lightning-AI/lightning/issues/2534#issuecomment-674582085
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def get_argparse_groups(parser):
	groups = {}
	for group in parser._action_groups:
		group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
		groups[group.title] = argparse.Namespace(**group_dict)
	return groups

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정

seed_everything(20)


if __name__ == '__main__':

	# throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
	base_parser = ArgumentParser(add_help=False)
	parser = ArgumentParser()
	for parser_ in (base_parser, parser):
		parser_.add_argument("--mode", default= "regen-joint-training", choices=["regen-joint-freeze","score-only", "denoiser-only", "regen-freeze-denoiser", "regen-joint-training"],
			help="score-only calls the ScoreModel class, \
				  denoiser-only calls the DiscriminativeModel class, \
				  regen-... calls the StochasticRegenerationModel class with the following options: \
				  	- regen-freeze-denoiser will freeze the denoiser, make sure to call a pretrained model \
					- regen-joint-training will not freeze the denoiser and consequently will train jointly the denoiser and score model")
		parser_.add_argument("--backbone_denoiser", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp_crossatt")
		parser_.add_argument("--pretrained_denoiser", default=None, help="checkpoint for denoiser") 
		parser_.add_argument("--backbone_score", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp_crossatt")
		parser_.add_argument("--pretrained_score", default= None, help="checkpoint for score") 
		parser_.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
		parser_.add_argument("--logstdout", action="store_true", help="Whether to print the stdout in a separate file")
		parser_.add_argument("--discriminatively", action="store_true", help="Train the backbone as a discriminative model instead")
	temp_args, _ = base_parser.parse_known_args()
	if "regen" in temp_args.mode:
		model_cls = StochasticRegenerationModel
	

	backbone_cls_denoiser = BackboneRegistry.get_by_name(temp_args.backbone_denoiser) if temp_args.backbone_denoiser != "none" else None
	backbone_cls_score = BackboneRegistry.get_by_name(temp_args.backbone_score) if temp_args.backbone_score != "none" else None

	parser = pl.Trainer.add_argparse_args(parser)
	model_cls.add_argparse_args(
		parser.add_argument_group(model_cls.__name__, description=model_cls.__name__))
			
	if temp_args.backbone_denoiser != "none":
		backbone_cls_denoiser.add_argparse_args(
			parser.add_argument_group("BackboneDenoiser", description=backbone_cls_denoiser.__name__))
	else:
		parser.add_argument_group("BackboneDenoiser", description="none")

	if temp_args.backbone_score != "none":
		backbone_cls_score.add_argparse_args(
			parser.add_argument_group("BackboneScore", description=backbone_cls_score.__name__))
	else:
		parser.add_argument_group("BackboneScore", description="none")
	
	

	# Add data module args
	data_module_cls = SpecsDataModule
	data_module_cls.add_argparse_args(
		parser.add_argument_group("DataModule", description=data_module_cls.__name__))
	args = parser.parse_args()
	arg_groups = get_argparse_groups(parser)

	# Initialize logger, trainer, model, datamodule
	if "regen" in temp_args.mode:
		model = model_cls(
			mode=args.mode, backbone_denoiser=args.backbone_denoiser, backbone_score=args.backbone_score,  data_module_cls=data_module_cls,
			**{
				**vars(arg_groups['StochasticRegenerationModel']),
				**vars(arg_groups['BackboneDenoiser']),
				**vars(arg_groups['BackboneScore']),
				**vars(arg_groups['DataModule'])
			},
			nolog=args.nolog
		)
		if temp_args.pretrained_denoiser is not None:
			model.load_denoiser_model(temp_args.pretrained_denoiser)
		if temp_args.pretrained_score is not None:
			#model.load_score_model(torch.load(temp_args.pretrained_score))
			model.load_score_model((temp_args.pretrained_score))
		data_tag = model.data_module.base_dir.strip().split("/")[-3] if model.data_module.format == "whamr" else model.data_module.base_dir.strip().split("/")[-1] 
		logging_name = f"mode={model.mode}_score={temp_args.backbone_score}_data={model.data_module.format}_ch={model.data_module.spatial_channels}"
		
	
	logger = TensorBoardLogger(save_dir=f"./.logs/", name=logging_name, flush_secs=30) if not args.nolog else None


	# Callbacks
	callbacks = []
	callbacks.append(TQDMProgressBar(refresh_rate=50))
	if not args.nolog:
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), 
			save_last=True, save_top_k=1, monitor="valid_loss", filename='{epoch}'))
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), 
			save_top_k=1, monitor="ValidationPESQ", mode="max", filename='{epoch}-{pesq:.2f}'))
		callbacks.append(CheckpointEveryNSteps(save_step_frequency=30000))

	# Initialize the Trainer and the DataModule
	trainer = pl.Trainer.from_argparse_args(
		arg_groups['pl.Trainer'],
		strategy=DDPStrategy(find_unused_parameters=True), #strategy = "ddp",
		accelerator = 'gpu',
		devices = torch.cuda.device_count(), #can set to 1 for sigle gpu training
		logger=logger,
		log_every_n_steps=10, num_sanity_val_steps=0, 
		callbacks=callbacks,
		max_epochs=40,
		val_check_interval= 0.5
	)
	
	
	trainer.fit(model)