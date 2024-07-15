from math import ceil
import warnings

import matplotlib.pyplot as plt
from inspect import isfunction
import math
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import wandb
import time
import os
import numpy as np
from torch.nn import MultiheadAttention
import torch.nn as nn
from torch.nn import functional as F
from torch import einsum
from einops import rearrange, repeat

#from .diffusion_utils import checkpoint

import random

from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.graphics import visualize_example, visualize_one
from sgmse.util.other import pad_spec, si_sdr_torch
#from sgmse.backbones.ncsnpp import visualFrontend
VIS_EPOCHS = 1 

#torch.autograd.set_detect_anomaly(True)

class StochasticRegenerationModel(pl.LightningModule):
	def __init__(self,
		backbone_denoiser: str, backbone_score: str, 
		lr: float = 1e-4, ema_decay: float = 0.999,
		t_eps: float = 3e-2, nolog: bool = False, num_eval_files: int = 50,
		loss_type_denoiser: str = "none", loss_type_angle: str = 'mae',loss_type_score: str = 'mse', data_module_cls = None, 
		mode = "regen-joint-training", condition = "post_denoiser",
		**kwargs 
	):
		"""
		Create a new ScoreModel.
		Args:
			backbone: The underlying backbone DNN that serves as a score-based model.
				Must have an output dimensionality equal to the input dimensionality.
			lr: The learning rate of the optimizer. (1e-4 by default).
			ema_decay: The decay constant of the parameter EMA (0.999 by default).
			t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
			reduce_mean: If `True`, average the loss across data dimensions.
				Otherwise sum the loss across data dimensions.
		"""
		super().__init__()
		# Initialize Backbone DNN
		kwargs_denoiser = kwargs
		kwargs_denoiser.update(input_channels=2)
		kwargs_denoiser.update(discriminative=True)
		self.denoiser_net = BackboneRegistry.get_by_name(backbone_denoiser)(**kwargs) if backbone_denoiser != "none" else None

		kwargs.update(input_channels=(6 if condition == "both" else 4))
		kwargs_denoiser.update(discriminative=False)
		self.score_net = BackboneRegistry.get_by_name(backbone_score)(**kwargs) if backbone_score != "none" else None
		self.backbone_denoiser = backbone_denoiser
		self.backbone_score = backbone_score
		
		self.t_eps = t_eps

		self.lr = lr
		self.ema_decay = ema_decay
		self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
		self._error_loading_ema = False

		self.loss_type_denoiser = loss_type_denoiser
		self.loss_type_score = loss_type_score

		if "weighting_denoiser_to_score" in kwargs.keys():
			self.weighting_denoiser_to_score = kwargs["weighting_denoiser_to_score"]
		else:
			self.weighting_denoiser_to_score = .5
		self.condition = condition
		self.mode = mode
		self.configure_losses()
		

		self.num_eval_files = num_eval_files
		self.save_hyperparameters(ignore=['nolog'])
		self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
		self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
		self.nolog = nolog

		self.sigma_min=1e-4
		

		

	@staticmethod
	def add_argparse_args(parser):
		parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
		parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
		parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
		parser.add_argument("--num_eval_files", type=int, default=100, help="Number of files for speech enhancement performance evaluation during training.")
		parser.add_argument("--loss_type_denoiser", type=str, default="mse", choices=("none", "mse", "mae", "sisdr", "mse_cplx+mag", "mse_time+mag"), help="The type of loss function to use.")
		parser.add_argument("--loss_type_score", type=str, default="mse", choices=("none", "mse", "mae"), help="The type of loss function to use.")
		parser.add_argument("--loss_type_angle", type=str, default="mae", choices=("none", "mse", "mae"), help="The type of loss function to use.")        
		parser.add_argument("--weighting_denoiser_to_score", type=float, default=0.5, help="a, as in L = a * L_denoiser + (1-a) * .")
		parser.add_argument("--condition", default="post_denoiser", choices=["noisy", "post_denoiser", "both"])
		parser.add_argument("--spatial_channels", type=int, default=1)
		return parser

	def configure_losses(self):
		# Score Loss
		if self.loss_type_score == "mse":
			self.loss_fn_score = lambda err, vec: self._reduce_op(torch.square(torch.abs(err - vec)))
		else:
			raise NotImplementedError
		
		# Denoiser Loss
		if self.loss_type_denoiser == "mse":
			self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.square(torch.abs(x - y)))
		elif self.loss_type_denoiser == "mae":
			self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.abs(x - y))
		elif self.loss_type_denoiser == "none":
			self.loss_fn_denoiser = None
		else:
			raise NotImplementedError

		



	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		return optimizer

	def optimizer_step(self, *args, **kwargs):
		# Method overridden so that the EMA params are updated after each optimizer step
		super().optimizer_step(*args, **kwargs)
		self.ema.update(self.parameters())

	def load_denoiser_model(self, checkpoint):
		self.denoiser_net = DiscriminativeModel.load_from_checkpoint(checkpoint).dnn
		print("denoiser loaded")
		

	def load_score_model(self, checkpoint):
		self.score_net = ScoreModel.load_from_checkpoint(checkpoint).dnn

	# on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
	def on_load_checkpoint(self, checkpoint):
		ema = checkpoint.get('ema', None)
		if ema is not None:
			self.ema.load_state_dict(checkpoint['ema'])
		else:
			self._error_loading_ema = True
			warnings.warn("EMA state_dict not found in checkpoint!")

	def on_save_checkpoint(self, checkpoint):
		checkpoint['ema'] = self.ema.state_dict()

	def train(self, mode=True, no_ema=False):
		res = super().train(mode)  # call the standard `train` method with the given mode
		if not self._error_loading_ema:
			if mode == False and not no_ema:
				# eval
				self.ema.store(self.parameters())        # store current params in EMA
				self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
			else:
				# train
				if self.ema.collected_params is not None:
					self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
		return res

	def eval(self, no_ema=False):
		return self.train(False, no_ema=no_ema)

	def _loss(self, err, y_denoised, x, vec):
		if self.mode == "regen-joint-freeze":
			loss_denoiser = self.loss_fn_denoiser(y_denoised, x) if self.loss_type_denoiser != "none" else None
			return loss_denoiser, None, None

		else:
			loss_score = self.loss_fn_score(err,vec) if self.loss_type_score != "none" else None #mse
			loss_denoiser = self.loss_fn_denoiser(y_denoised, x) if self.loss_type_denoiser != "none" else None

			loss = self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score  
			
		return loss, loss_score, loss_denoiser

	def _weighted_mean(self, x, w):
		return torch.mean(x * w)

	
	def forward_score(self, x, t, score_conditioning,  context=None, **kwargs):
		dnn_input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t
		score = self.score_net(x = dnn_input,context = context, time_cond = t) #,context= context)
		if len(score)==2:
			score = -score[0]
		return score

	def forward_denoiser(self, y, context=None, **kwargs):
		x_hat, context = self.denoiser_net(y, context)
		return x_hat, context
	
	def sample_x(self, mean, sigma):
		eps = torch.randn_like(mean)
		return mean+eps*sigma


	def _step(self, batch, batch_idx):
		x, y, visualFeatures = batch
		y_trans = y
		# Denoising step
		y_denoised, context = self.forward_denoiser(y_trans, context = visualFeatures) # y noisy speech + visual feature

		if self.mode == "regen-joint-freeze":
			loss = self._loss(None, y_denoised, x, None)
			return loss

		# CFM part
		t1 = torch.rand([x.shape[0],1,1,1], device=x.device)* (1- self.t_eps) + self.t_eps #original code version self.t_eps/2
		
		#1) Start with Gaussian
		#z = torch.randn_like(x) 
		#y1 = (1 - (1 - self.sigma_min) * t1) * z + t1 * x
		#vec = x - (1 - self.sigma_min) * z

		#2) Start with y_denoised
		goal = torch.randn_like(x) * self.sigma_min + x
		z = torch.randn_like(y_denoised) * self.sigma_min + y_denoised
		y1 = (1 - t1) * z + t1 * goal

		y1 = torch.randn_like(y1) * self.sigma_min + y1
		vec = goal-z

		# Score estimation
		t1 = t1.squeeze()
		if x.shape[0]==1:
			t1 = t1.unsqueeze(0)


		score_conditioning = [y_denoised] 
		score_1 = self.forward_score(y1, t1, score_conditioning, context=visualFeatures)
		
		err1 = score_1
		
		loss, loss_score, loss_denoiser = self._loss(err1, y_denoised, x, vec)
		
		return loss, loss_score, loss_denoiser

	def training_step(self, batch, batch_idx):
		loss, loss_score, loss_denoiser = self._step(batch, batch_idx)
		self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
		if self.mode =="regen-joint-freeze":
			return loss
		self.log('train_loss_score', loss_score, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
		if loss_denoiser is not None:
			self.log('train_loss_denoiser', loss_denoiser, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
		return loss

	def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
		loss, loss_score, loss_denoiser = self._step(batch, batch_idx)

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
		if not self.mode == "regen-joint-freeze":
			self.log('valid_loss_score', loss_score, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
		if loss_denoiser is not None:
			self.log('valid_loss_denoiser', loss_denoiser, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files
			if self.current_epoch %2 ==0 and self.current_epoch !=0:
				num_eval_files =100
			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
			print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
			print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
			print(f"PESQ Denoiser at epoch {self.current_epoch} : {y_den[0]:.2f}")
			print(f"SISDR Denoiser at epoch {self.current_epoch} : {y_den[1]:.1f}")
			print(f"ESTOI Denoiser at epoch {self.current_epoch} : {y_den[2]:.2f}")
			print('__________________________________________________________________')
			
			self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True, sync_dist=True)
			self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True, sync_dist=True)
			self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True, sync_dist=True)

			if audio is not None:
				y_list, x_hat_list, x_list = audio
				for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
					if self.current_epoch == 0:
						self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(-1),self.current_epoch, sample_rate=sr)
						self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(-1),self.current_epoch, sample_rate=sr)
					self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(-1),self.current_epoch, sample_rate=sr)
			'''
			if spec is not None:
				figures = []
				y_stft_list, x_hat_stft_list, x_stft_list = spec
				for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
					figures.append(
						visualize_example(
						torch.abs(y_stft), 
						torch.abs(x_hat_stft), 
						torch.abs(x_stft), return_fig=True))
				self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures) #, sync_dist=True)
			'''

		return loss

	def to(self, *args, **kwargs):
		self.ema.to(*args, **kwargs)
		return super().to(*args, **kwargs)

	def train_dataloader(self):
		return self.data_module.train_dataloader()

	def val_dataloader(self):
		return self.data_module.val_dataloader()

	def test_dataloader(self):
		return self.data_module.test_dataloader()

	def setup(self, stage=None):
		return self.data_module.setup(stage=stage)

	def to_audio(self, spec, length=None):
		return self._istft(self._backward_transform(spec), length)

	def _forward_transform(self, spec):
		return self.data_module.spec_fwd(spec)

	def _backward_transform(self, spec):
		return self.data_module.spec_back(spec)

	def _stft(self, sig):
		return self.data_module.stft(sig)

	def _istft(self, spec, length=None):
		return self.data_module.istft(spec, length)

	def enhance(self, y,context, timeit=False,
		return_stft=False,
		**kwargs
	):

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.
		"""
		start = time.time()
		T_orig = y.size(1)
		norm_factor = y.abs().max().item()
		y = y / norm_factor
		Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
		Y, num_pad = pad_spec(Y)
		with torch.no_grad():
			if self.denoiser_net is not None:
				y_trans = Y
				Y_denoised, cont = self.forward_denoiser(y_trans, context=context)
			else:
				Y_denoised = None
			
			if not self.mode == "regen-joint-freeze":
				t_span = torch.linspace(self.t_eps, 1 , 1+ 1, device=Y_denoised.device)
				t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
				sol = []
				vector=[]
				steps = 1 # 1 step inference for CFM

				x = torch.randn_like(Y_denoised) * self.sigma_min + Y_denoised
				
				while steps <= len(t_span) - 1:
					dphi_dt = self.forward_score(x, torch.tensor([t]).cuda(), [Y_denoised], context=context)
					x = x + dt * dphi_dt
					t = t + dt
					sol.append(x)
					vector.append(dphi_dt)
					if steps < len(t_span) - 1:
						dt = t_span[steps + 1] - t
					steps += 1
				sample = sol[-1]

				
				if return_stft:
					tot = sample.shape[-1]
					sample = sample[:,:,:,: tot-num_pad]
					Y = Y[:,:,:,: tot-num_pad]
					Y_denoised = Y_denoised[:,:,:,: tot-num_pad]
					return sample.squeeze(), Y.squeeze(), Y_denoised.squeeze(), T_orig, norm_factor
			else:
				sample = Y_denoised


		x_hat = self.to_audio(sample.squeeze(), T_orig)
		x_hat = x_hat * norm_factor
		x_hat = x_hat.squeeze().cpu()

		Y_denoised = self.to_audio(Y_denoised.squeeze(), T_orig)
		Y_denoised = Y_denoised * norm_factor
		Y_denoised = Y_denoised.squeeze().cpu()

		end = time.time()
		if timeit:
			sr = 16000
			rtf = (end-start)/(len(x_hat)/sr)
			return x_hat, nfe, rtf
		else:
			return x_hat , Y_denoised
