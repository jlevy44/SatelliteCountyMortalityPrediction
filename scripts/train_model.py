import torch
import glob
from functools import reduce
import pandas as pd, numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models
from torch import nn
from torch.nn import PoissonNLLLoss as PoissonLoss
import math
from torch.autograd import Variable
import copy
from sklearn.metrics import r2_score, mean_absolute_error
import argparse
import pickle
import pysnooper

def PoissonLossOld(y_pred, y_true):
	"""Custom loss function for Poisson model."""
	loss=torch.mean(torch.exp(y_pred)-y_true*y_pred)
	return loss

def generate_transformers(image_size=224, mean=[], std=[], include_jitter=True):

	train_transform = transforms.Compose([
		transforms.Resize(256)]+\
		#([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)] if include_jitter else [])+\
		[transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomVerticalFlip(p=0.5),
		transforms.RandomRotation(30),
		transforms.RandomResizedCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean if mean else [0.5, 0.5, 0.5],
							 std if std else [0.1, 0.1, 0.1])
	])
	val_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean if mean else [0.5, 0.5, 0.5],
							 std if std else [0.1, 0.1, 0.1])
	])
	normalization_transform = transforms.Compose([transforms.Resize(256),
												  transforms.CenterCrop(image_size),
										transforms.ToTensor()])
	return {'train': train_transform, 'val': val_transform, 'test': val_transform, 'norm':normalization_transform}

class SklearnDeep:
	def __init__(self, model,
						n_epoch=300,
						validation_dataloader=None,
						optimizer_opts=dict(name='adam',lr=1e-3,weight_decay=1e-4),
						scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2),
						loss_fn='ce',
						use_covariates=False,
						checkpoint_dir='checkpoints',
						save_load_dict=False,
						eval_test_during_training=False,
						test_dataloader=None):
		self.model = model
		optimizers = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}
		loss_functions = {'bce':nn.BCELoss(), 'ce':nn.CrossEntropyLoss(), 'mse':nn.MSELoss(), 'poisson':PoissonLoss(log_input=False)}
		if 'name' not in list(optimizer_opts.keys()):
			optimizer_opts['name']='adam'
		self.optimizer = optimizers[optimizer_opts.pop('name')](self.model.parameters(),**optimizer_opts)
		self.scheduler = Scheduler(optimizer=self.optimizer,opts=scheduler_opts)
		self.n_epoch = n_epoch
		self.validation_dataloader = validation_dataloader
		self.loss_fn = loss_functions[loss_fn]
		self.use_covariates=use_covariates
		self.save_load_dict=save_load_dict
		self.checkpoint_dir=checkpoint_dir
		self.eval_test_during_training=eval_test_during_training
		os.makedirs(checkpoint_dir,exist_ok=True)
		self.test_dataloader=test_dataloader

	def calc_loss(self, y_pred, y_true):
		return self.loss_fn(y_pred, y_true)

	def train_loop(self, train_dataloder):
		self.model.train(True)
		n_batch=len(train_dataloder.dataset)//train_dataloder.batch_size
		running_loss = 0.
		for i, (X,y_true,covar) in enumerate(train_dataloder):
			# X = Variable(batch[0], requires_grad=True)
			# y_true = Variable(batch[1])
			if torch.cuda.is_available():
				X,y_true=X.cuda(),y_true.cuda()
				covar=covar.cuda()
			if self.use_covariates:
				y_pred = self.model(X,covar)
			else:
				y_pred = self.model(X)
			loss = self.calc_loss(y_pred,y_true)
			train_loss=loss.item()
			print('Epoch {} Batch [{}/{}] Train Loss {}'.format(self.epoch,i,n_batch,train_loss))
			running_loss += train_loss
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		self.scheduler.step()
		running_loss/=(i+1)
		return running_loss

	def val_loop(self, val_dataloader):
		self.model.train(False)
		n_batch=len(val_dataloader.dataset)//val_dataloader.batch_size
		running_loss = 0.
		Y={'true':[],'pred':[]}
		with torch.no_grad():
			for i, (X,y_true,covar) in enumerate(val_dataloader):
				# X = Variable(batch[0],requires_grad=False)
				# y_true = Variable(batch[1])
				if torch.cuda.is_available():
					X,y_true=X.cuda(),y_true.cuda()
					covar=covar.cuda()
				if self.use_covariates:
					y_pred = self.model(X,covar)
				else:
					y_pred = self.model(X)
				loss = self.calc_loss(y_pred,y_true)
				val_loss=loss.item()
				print('Epoch {} Batch [{}/{}] Val Loss {}'.format(self.epoch,i,n_batch,val_loss))
				running_loss += val_loss
				Y['true'].extend(y_true.detach().cpu().numpy().flatten().tolist())
				Y['pred'].extend(y_pred.detach().cpu().numpy().flatten().tolist())
			print(pd.DataFrame.from_dict(Y))
		print('Epoch {}: Val R2: {}, Val MAE: {}'.format(self.epoch,r2_score(Y['true'],Y['pred']), mean_absolute_error(Y['true'],Y['pred'])))
		running_loss/=(i+1)
		return running_loss

	def save_model(self,epoch,test_dataloader=None):
		save_item=self.model if not self.save_load_dict else self.model.state_dict()
		torch.save(save_item,os.path.join(self.checkpoint_dir,'checkpoint.{}.pkl'.format(epoch)))
		if self.eval_test_during_training:
			y_true=self.test_dataloader.dataset.y#datasets.loc[datasets['Set']=='test']['Mortality'].values.flatten()
			y_pred=self.predict(self.test_dataloader).flatten()

			y_true_val=self.val_dataloader.dataset.y#datasets.loc[datasets['Set']=='test']['Mortality'].values.flatten()
			y_pred_val=self.predict(self.val_dataloader).flatten()

			results=dict(val=dict(y_pred=y_pred_val,y_true=y_true_val),test=dict(y_pred=y_pred,y_true=y_true))

			torch.save(results,os.path.join(self.checkpoint_dir,'predictions_{}.pkl'.format(epoch)))

	def test_loop(self, test_dataloader):
		self.model.train(False)
		y_pred = []
		running_loss = 0.
		with torch.no_grad():
			for i, (X,y_true,covar) in enumerate(test_dataloader):
				# X = Variable(batch[0],requires_grad=False)
				if torch.cuda.is_available():
					X=X.cuda()
					covar=covar.cuda()
				y_pred.append((self.model(X) if not self.use_covariates else self.model(X,covar)).detach().cpu())
			y_pred = torch.cat(y_pred,0).numpy()
		return y_pred

	def fit(self, train_dataloader, verbose=True, print_every=1, save_model=True):
		train_losses = []
		val_losses = []
		for epoch in range(self.n_epoch):
			self.epoch=epoch
			train_loss = self.train_loop(train_dataloader)
			train_losses.append(train_loss)
			val_loss = self.val_loop(self.validation_dataloader)
			val_losses.append(val_loss)
			if verbose and not (epoch % print_every):
				print("Epoch {}: Train Loss {}, Val Loss {}".format(epoch,train_loss,val_loss))
			if val_loss <= min(val_losses) and save_model:
				min_val_loss = val_loss
				best_epoch = epoch
				best_model = copy.deepcopy((self.model if not self.save_load_dict else self.model.state_dict()))
				self.save_model(epoch)
		if save_model:
			print("Loading best model at epoch {}".format(best_epoch))
			if self.save_load_dict:
				self.model.load_state_dict(best_model)
			else:
				self.model = best_model
		self.train_losses = train_losses
		self.val_losses = val_losses
		return self, min_val_loss, best_epoch

	def plot_train_val_curves(self):
		plt.figure()
		sns.lineplot('epoch','value',hue='variable',
					 data=pd.DataFrame(np.vstack((np.arange(len(self.train_losses)),self.train_losses,self.val_losses)).T,
									   columns=['epoch','train','val']).melt(id_vars=['epoch'],value_vars=['train','val']))

	def predict(self, test_dataloader):
		y_pred = self.test_loop(test_dataloader)
		return y_pred

	def fit_predict(self, train_dataloader, test_dataloader):
		return self.fit(train_dataloader)[0].predict(test_dataloader)

class CosineAnnealingWithRestartsLR(torch.optim.lr_scheduler._LRScheduler):
	r"""Set the learning rate of each parameter group using a cosine annealing
	schedule, where :math:`\eta_{max}` is set to the initial lr and
	:math:`T_{cur}` is the number of epochs since the last restart in SGDR:
	 .. math::
		 \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
		\cos(\frac{T_{cur}}{T_{max}}\pi))
	 When last_epoch=-1, sets initial lr as lr.
	 It has been proposed in
	`SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implements
	the cosine annealing part of SGDR, the restarts and number of iterations multiplier.
	 Args:
		optimizer (Optimizer): Wrapped optimizer.
		T_max (int): Maximum number of iterations.
		T_mult (float): Multiply T_max by this number after each restart. Default: 1.
		eta_min (float): Minimum learning rate. Default: 0.
		last_epoch (int): The index of last epoch. Default: -1.
	 .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
		https://arxiv.org/abs/1608.03983
	"""
	def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1., alpha_decay=1.0):
		self.T_max = T_max
		self.T_mult = T_mult
		self.restart_every = T_max
		self.eta_min = eta_min
		self.restarts = 0
		self.restarted_at = 0
		self.alpha = alpha_decay
		super().__init__(optimizer, last_epoch)

	def restart(self):
		self.restarts += 1
		self.restart_every = int(round(self.restart_every * self.T_mult))
		self.restarted_at = self.last_epoch

	def cosine(self, base_lr):
		return self.eta_min + self.alpha**self.restarts * (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2

	@property
	def step_n(self):
		return self.last_epoch - self.restarted_at

	def get_lr(self):
		if self.step_n >= self.restart_every:
			self.restart()
		return [self.cosine(base_lr) for base_lr in self.base_lrs]

class Scheduler:
	def __init__(self, optimizer=None, opts=dict(scheduler='null',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2)):
		self.schedulers = {'exp':(lambda optimizer: ExponentialLR(optimizer, opts["lr_scheduler_decay"])),
							'null':(lambda optimizer: None),
							'warm_restarts':(lambda optimizer: CosineAnnealingWithRestartsLR(optimizer, T_max=opts['T_max'], eta_min=opts['eta_min'], last_epoch=-1, T_mult=opts['T_mult']))}
		self.scheduler_step_fn = {'exp':(lambda scheduler: scheduler.step()),
								  'warm_restarts':(lambda scheduler: scheduler.step()),
								  'null':(lambda scheduler: None)}
		self.initial_lr = optimizer.param_groups[0]['lr']
		self.scheduler_choice = opts['scheduler']
		self.scheduler = self.schedulers[self.scheduler_choice](optimizer) if optimizer is not None else None

	def step(self):
		self.scheduler_step_fn[self.scheduler_choice](self.scheduler)

	def get_lr(self):
		lr = (self.initial_lr if self.scheduler_choice == 'null' else self.scheduler.optimizer.param_groups[0]['lr'])
		return lr

def normalize(dataloader):
	mean = torch.tensor([0.,0.,0.])
	std = torch.tensor([0.,0.,0.])
	if torch.cuda.is_available():
		mean,std=mean.cuda(),std.cuda()

	with torch.no_grad():
		for i,(X,_,_) in enumerate(dataloader):
			if torch.cuda.is_available():
				X=X.cuda()
			mean += torch.mean(X, (0,2,3))
			std += torch.std(X, (0,2,3))

	N=i+1
	mean,std=(np.array([mean.cpu().numpy(),std.cpu().numpy()])/float(N)).tolist()
	#mean = (np.array(means).mean(axis=0)).tolist()
	#std = (np.array(stds).mean(axis=0)).tolist()
	return dict(mean=mean,std=std)

class BasicImageSet(Dataset):
	def __init__(self, img_df, Set, transformers, output_col='Mortality', label_noise_factor=0., add_idx=False,covariates=None):
		setname=Set
		if Set=='norm':
			setname='train'
		self.df=img_df.loc[img_df['Set']==setname]
		self.Set=Set
		self.transformer=transformers[Set]
		self.images=self.df['Image'].values
		self.y=self.df[output_col].values
		self.length=self.df.shape[0]
		self.label_noise_factor=label_noise_factor
		self.add_idx=add_idx
		self.counties=self.df['County'].values
		print(self.counties)
		self.covariates=covariates

	def __getitem__(self, i):
		img=Image.open(self.images[i]).convert('RGB')
		img=self.transformer(img)
		y=torch.FloatTensor([self.y[i]])
		covar=torch.FloatTensor([self.covariates.loc[self.counties[i]].values])
		if self.label_noise_factor and self.Set=='train':
			y=y+torch.rand(1,1,dtype=torch.float)*self.label_noise_factor
		if self.add_idx:
			return torch.tensor(np.array([i]).reshape(-1,1)),img,y,covar
		else:
			return img, y, covar

	def __len__(self):
		return self.length

class GridImageSet(Dataset):
	def __init__(self, img_df, Set, transformers, outcome_col='Mortality', label_noise_factor=0., covariates=None):
		raise NotImplementedError("To be implemented")
		self.df=img_df
		self.Set=Set
		self.df['CountySchool']=self.df['County'].map(lambda x: '{}_'.format(x))+self.df['School'].astype(str)
		self.df=self.df.loc[img_df['Set']==Set]
		self.county_schools=self.df['CountySchool'].unique()
		print(self.county_schools)
		self.transformer=transformers[Set]
		self.images={name:dff['Image'].values for name, dff in self.df.groupby('CountySchool')}
		self.y=self.df.groupby('CountySchool')[outcome_col].mean().values
		print(self.y)
		self.length=len(self.county_schools)
		self.label_noise_factor=label_noise_factor
		self.counties=self.df['County'].values
		self.covariates=covariates

	def __getitem__(self, i):
		images = self.images[self.county_schools[i]]
		imgs=[self.transformer(Image.open(img).convert('RGB')).unsqueeze(0) for img in images]
		imgs=torch.cat(imgs,0)
		y=torch.FloatTensor([self.y[i]])
		covar=torch.FloatTensor([self.covariates.loc[self.counties[i]].values])
		if self.label_noise_factor and self.Set=='train':
			y+=torch.rand(1,1,dtype=torch.float)*self.label_noise_factor
		return imgs, y, covar

	def __len__(self):
		return self.length

class CovarModel(nn.Module):
	def __init__(self,feature_extractor,n_covar,n_classes=1,bilinear=True, dropout_p=0.2):
		super().__init__()
		self.model,self.image_features=feature_extractor,copy.deepcopy(feature_extractor.fc)
		self.model.fc=nn.Dropout(0.)
		self.relu=nn.ReLU()
		self.sigmoid=nn.Sigmoid()
		self.n_covar=n_covar
		self.bilinear=bilinear
		self.attention_weights=nn.ModuleList([nn.Linear(2*n_covar,n_covar),nn.Linear(2*n_covar,n_covar)])#[nn.Bilinear(n_covar,n_covar,n_covar),nn.Bilinear(n_covar,n_covar,n_covar)])
		self.covar_model=nn.Sequential(nn.Dropout(dropout_p),nn.Linear((n_covar+1)**2,n_classes)) #if not self.bilinear else nn.Bilinear(n_covar,n_covar,n_classes)

	def forward(self, X, covar):
		Z = self.model(X)
		covar=covar.squeeze(1)
		multi_modal_data=[self.relu(self.image_features(Z)),covar]
		multi_modal_data_tmp=[]
		cat_modal=torch.cat(multi_modal_data,1)
		for i in range(2):
			multi_modal_data_tmp.append(torch.sigmoid(self.attention_weights[i](cat_modal))*multi_modal_data[i])#cat_modal
		multi_modal_data=[torch.cat((multi_modal_data_tmp[i], torch.cuda.FloatTensor(multi_modal_data_tmp[i].shape[0], 1).fill_(1)), 1) for i in range(2)]
		Z = torch.bmm(multi_modal_data[0].unsqueeze(2), multi_modal_data[1].unsqueeze(1)).flatten(start_dim=1)#self.covar_model(torch.prod(multi_modal_data)) if not self.bilinear else self.covar_model(*multi_modal_data)
		return self.covar_model(Z)#Y
		# Z=self.model(X)
		# # Z=self.fc[0](Z)
		# # print(Z.shape,covar.shape)
		# Z=torch.cat([Z,covar.squeeze(1)],1)
		#Y=self.fc(Z)

class MLP(nn.Module): # add latent space extraction, and spits out csv line of SQL as text for UMAP
	def __init__(self, n_input, hidden_topology, dropout_p, n_outputs=1, binary=False, softmax=False, relu_out=True):
		super(MLP,self).__init__()
		self.hidden_topology=hidden_topology
		self.topology = [n_input]+hidden_topology+[n_outputs]
		layers = [nn.Linear(self.topology[i],self.topology[i+1]) for i in range(len(self.topology)-2)]
		for layer in layers:
			torch.nn.init.xavier_uniform_(layer.weight)
		self.layers = [nn.Sequential(layer,nn.ReLU(),nn.Dropout(p=dropout_p)) for layer in layers]
		self.output_layer = nn.Linear(self.topology[-2],self.topology[-1])
		torch.nn.init.xavier_uniform_(self.output_layer.weight)
		if binary:
			output_transform = nn.Sigmoid()
		elif softmax:
			output_transform = nn.Softmax()
		elif relu_out:
			output_transform = nn.ReLU()
		else:
			output_transform = nn.Dropout(p=0.)
		self.layers.append(nn.Sequential(self.output_layer,output_transform))
		self.mlp = nn.Sequential(*self.layers)

class GridNetwork(nn.Module):
	def __init__(self, feature_extractor_model, kernel1, kernel2, grid_length, n_outputs, n_hidden, averaged_effect=False, dropout_p=0.):
		super(GridNetwork,self).__init__()
		raise NotImplementedError("To be implemented")
		self.input_channels=n_hidden#feature_extractor_model.fc.hidden_topology[0]
		#print(feature_extractor_model.fc)
		#print(nn.Sequential(*list(feature_extractor_model.fc.children())[:-1])[0])
		feature_extractor_model.fc, prediction_layer=nn.Sequential(*list(feature_extractor_model.fc.children())[:-1])[0], (list(feature_extractor_model.fc.children())[-1] if averaged_effect else None)  #if not averaged_effect else feature_extractor_model.fc
		self.feature_extraction=feature_extractor_model
		self.grid_length=grid_length
		self.name='GridNetwork'
		if not averaged_effect:
			self.conv1=nn.Conv2d(self.input_channels,20,kernel1)
			torch.nn.init.xavier_uniform(self.conv1.weight)
			grid_length=grid_length-kernel1+1
			self.conv2=nn.Conv2d(20,5,kernel2)
			torch.nn.init.xavier_uniform(self.conv2.weight)
			grid_length=grid_length-kernel2+1
			total_size=(grid_length**2)*5
			self.attention = nn.Sequential(self.conv1,nn.ReLU(),self.conv2,nn.ReLU())
			self.fc=MLP(total_size,[1000],dropout_p,1).mlp
		else:
			self.fc = prediction_layer
		self.averaged_effect = averaged_effect

	def forward(self, x):
		x=x.squeeze(0)
		x=self.feature_extraction(x)
		if not self.averaged_effect:
			x=x.view(self.grid_length,self.grid_length,self.input_channels).unsqueeze(0).transpose(3,1).transpose(3,2)
			x=self.attention(x).flatten()
			x=self.fc(x)
		else:
			x=self.fc(torch.mean(x,dim=0))#torch.mean(self.fc(x),dim=0)
		return x

def generate_model(pretrained=False, num_classes=1, n_hidden=1000, architecture='resnet34', dropout_p=0., n_covar=0):
	model = getattr(models, architecture)(pretrained=pretrained)
	num_ftrs = model.fc.in_features
	linear_layer = MLP(num_ftrs,[n_hidden],dropout_p,num_classes if not n_covar else n_covar).mlp # +n_covar
	model.fc = linear_layer
	model.name='ImageNetwork'
	if n_covar:
		model=CovarModel(model,n_covar)
	return model

def generate_grid_model(feature_model, kernel1, kernel2, grid_length, n_outputs, n_hidden=1000, averaged=False):
	return GridNetwork(feature_model, kernel1, kernel2, grid_length, n_outputs, n_hidden, averaged_effect=averaged)

def train_test_split2(df,p=0.8,stratify_col='Mortality_binned'):
	np.random.seed(42)
	df=df.reset_index(drop=True)
	df_train=[]
	for name, dff in df.groupby(stratify_col):
		df_train.append(dff.sample(frac=p))
	df_train=pd.concat(df_train)
	df_test = df.loc[np.isin(df.index.values,df_train.index.values)==0]
	return df_train, df_test

#@pysnooper.snoop('main.log')
def main():
	# ADD CRF, LABEL NOISE, POISSON LOSS, AVG EFFECT instead grid, GAT?
	p = argparse.ArgumentParser()
	p.add_argument('--use_grid_model', action='store_true')
	p.add_argument('--lr', type=float, default=1e-3)
	p.add_argument('--averaged', action='store_true')
	p.add_argument('--model_save_loc', type=str)
	p.add_argument('--model_pretrain_loc', type=str)
	p.add_argument('--loss_fn', type=str, default='mse')
	p.add_argument('--architecture', type=str, default='resnet34')
	p.add_argument('--pretrain_imagenet', action='store_true')
	p.add_argument('--multiplicative_factor', type=float, default=1000.)
	p.add_argument('--extract_embeddings', action='store_true')
	p.add_argument('--embedding_set', type=str, default='test')
	p.add_argument('--dataset', type=str, default='data/training_datasets_original_new.csv')
	p.add_argument('--batch_size', type=int, default=64)
	p.add_argument('--num_workers', type=int, default=20)
	p.add_argument('--predict_mode', action='store_true')
	p.add_argument('--grid_len', type=int, default=7)
	p.add_argument('--effective_grid_len', type=int, default=7)
	p.add_argument('--dropout_p', type=float, default=0.)
	p.add_argument('--use_covariates', action='store_true')
	p.add_argument('--n_schools', type=int, default=4)
	p.add_argument('--save_load_dict', action='store_true')
	p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
	p.add_argument('--eval_test_during_training', action='store_true')
	p.add_argument('--train_residuals', action='store_true')


	args=p.parse_args()
	np.random.seed(42)
	grid_len=args.grid_len
	effective_grid_len=args.effective_grid_len
	remove_i=(grid_len-effective_grid_len)//2
	n_schools=args.n_schools
	use_covariates=args.use_covariates
	eval_test_during_training=args.eval_test_during_training
	#multiplicative_factor=1000.
	dataset=args.dataset
	image_dir="image_data/"
	norm_file='norm.pkl'
	predict_mode=args.predict_mode
	imagenet=args.pretrain_imagenet
	architecture=args.architecture
	use_grid_model=args.use_grid_model
	averaged=args.averaged
	model_save_loc=args.model_save_loc
	model_pretrain_loc=args.model_pretrain_loc
	loss_fn=args.loss_fn
	multiplicative_factor=args.multiplicative_factor
	extract_embeddings=args.extract_embeddings
	embedding_set=args.embedding_set
	batch_size=args.batch_size
	num_workers=args.num_workers
	n_schools_total=4
	checkpoint_dir=args.checkpoint_dir
	save_load_dict=args.save_load_dict
	train_residuals=args.train_residuals

	covariates=pd.read_csv("data/final_dataset_covariate_model.csv",index_col=1)[["Age","Adjusted_Race_1","Adjusted_Race_2","Adjusted_Race_4","Hispanic","Sex","Any_College_2015","income_2015","Region_1",'Region_2', 'Region_3', 'Region_4', 'Region_5', 'Region_6', 'Region_7','Region_8']]

	print(covariates)

	datasets_df=pd.read_csv(dataset,index_col=0)

	print(datasets_df)

	available_counties=[os.path.basename(d) for d in glob.glob(os.path.join(image_dir,'*')) if len(glob.glob(os.path.join(d,'*.png')))==n_schools_total*grid_len**2]

	datasets_df=datasets_df.reset_index(drop=True)
	datasets_df=datasets_df.loc[datasets_df['County'].isin(available_counties)]

	if train_residuals:
		datasets_df.loc[:,'Mortality']=datasets_df.loc[:,'Residual_Mortality']

	if multiplicative_factor>1.:
		datasets_df.loc[:,'Mortality']=datasets_df['Mortality']*multiplicative_factor

	print(datasets_df)

	datasets={}

	if 'Set' not in list(datasets_df):
		datasets['train'], datasets['test']= train_test_split2(datasets_df)

		datasets['train'], datasets['val'] = train_test_split2(datasets['train'])

		for dataset in datasets:
			datasets[dataset]['Set']=dataset
	else:
		for k in ['train','val','test']:
			datasets[k]=datasets_df[datasets_df['Set']==k]

	datasets = pd.concat(list(datasets.values()))

	feature_extractor_data = []

	for Set, dff in datasets.groupby('Set'):
		for i in range(dff.shape[0]):
			county_images=sorted(glob.glob(os.path.join(image_dir,dff.iloc[i]['County'],'*')))
			school=np.vectorize(lambda x: int(x.split('/')[-1].split('_')[3]))(county_images)
			image_number=np.vectorize(lambda x: int(x.split('_')[-1].split('.png')[0]))(county_images)
			dfff=pd.DataFrame(np.array(county_images)[:,np.newaxis],columns=['Image'])
			dfff['County']=dff.iloc[i]['County']
			dfff['Mortality']=dff.iloc[i]['Mortality']
			dfff['Set']=Set
			dfff['School']=school
			dfff['Idx']=image_number
			# if n_schools < dfff['School'].max()+1:
			dfff=dfff.loc[(dfff['School'].values+1)<=n_schools]
			feature_extractor_data.append(dfff)

	feature_extractor_data=pd.concat(feature_extractor_data)

	def get_remove_bool(pics):
		remove_bool=((pics>=7*remove_i)&(pics<(49-7*remove_i)))
		remove_bool2=(pics-pics//7*7)
		remove_bool2=((remove_bool) & ((remove_bool2 >= remove_i) & (remove_bool2 <= (6-remove_i))))
		return remove_bool2

	if remove_i>0:
		pics=np.vectorize(lambda x: int(x.split('_')[-1].replace('.png','')))(feature_extractor_data['Image'].values)
		remove_bool2=get_remove_bool(pics)
		print(np.arange(grid_len**2)[get_remove_bool(np.arange(grid_len**2))])
		feature_extractor_data=feature_extractor_data.loc[remove_bool2]

	feature_extractor_data=feature_extractor_data.sort_values(['Set','County','School','Idx'])#\

	transformers=generate_transformers(image_size=224, mean=[], std=[])

	norm_dataset=BasicImageSet(feature_extractor_data,'norm',transformers,covariates=covariates)

	if os.path.exists(norm_file):
		norm_opts=torch.load(norm_file)
	else:
		norm_opts=normalize(DataLoader(norm_dataset,batch_size=batch_size,num_workers=0,shuffle=True,drop_last=True))
		torch.save(norm_opts,norm_file)

	transformers=generate_transformers(image_size=224,**norm_opts)

	image_datasets={Set:BasicImageSet(feature_extractor_data,Set,transformers,covariates=covariates) for Set in ['train','val','test']}
	if use_grid_model:
		grid_datasets={Set:GridImageSet(feature_extractor_data,Set,transformers,covariates=covariates) for Set in ['train','val','test']}

	if model_pretrain_loc and os.path.exists(model_pretrain_loc):
		if save_load_dict:
			model=generate_model(pretrained=imagenet, architecture=architecture, dropout_p=args.dropout_p, n_covar=(covariates.shape[1] if use_covariates else 0))
			model.load_state_dict(torch.load(model_pretrain_loc))
		else:
			model=torch.load(model_pretrain_loc)
		pretrained=True
	else:
		model = generate_model(pretrained=imagenet, architecture=architecture, dropout_p=args.dropout_p, n_covar=(covariates.shape[1] if use_covariates else 0))
		pretrained=False
	dataloaders = {Set:DataLoader(image_datasets[Set],batch_size=batch_size,num_workers=num_workers,shuffle=(Set=='train')) for Set in ['train','val','test']}

	if use_grid_model:
		if model.name!='GridNetwork':
			model = generate_grid_model(model,3,3,grid_len,1, averaged=averaged)
		dataloaders = {Set:DataLoader(grid_datasets[Set], batch_size=1, num_workers=num_workers,shuffle=(Set=='train')) for Set in ['train','val','test']}

	if torch.cuda.is_available():
		model=model.cuda()

	trainer=SklearnDeep(model,
				n_epoch=50,
				validation_dataloader=dataloaders['val'],
				optimizer_opts=dict(name='adam',lr=args.lr,weight_decay=1e-4),
				scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2),
				loss_fn=loss_fn,
				use_covariates=use_covariates,
				checkpoint_dir=checkpoint_dir,
				save_load_dict=save_load_dict,
				eval_test_during_training=eval_test_during_training,
				test_dataloader=dataloaders['test'])#'mse')#'poisson')

	if not extract_embeddings:

		if not predict_mode:

			trainer.fit(dataloaders['train'])

			torch.save(trainer.model,model_save_loc)

		dataloaders['train']=DataLoader(image_datasets['train'] if not use_grid_model else grid_datasets['train'],batch_size=batch_size,num_workers=num_workers,shuffle=False)

		y_true=dataloaders['test'].dataset.y#datasets.loc[datasets['Set']=='test']['Mortality'].values.flatten()
		y_pred=trainer.predict(dataloaders['test']).flatten()

		y_true_val=dataloaders['val'].dataset.y#datasets.loc[datasets['Set']=='test']['Mortality'].values.flatten()
		y_pred_val=trainer.predict(dataloaders['val']).flatten()

		y_true_train=dataloaders['train'].dataset.y#datasets.loc[datasets['Set']=='test']['Mortality'].values.flatten()
		y_pred_train=trainer.predict(dataloaders['train']).flatten()

		results=dict(train=dict(y_pred=y_pred_train,y_true=y_true_train),val=dict(y_pred=y_pred_val,y_true=y_true_val),test=dict(y_pred=y_pred,y_true=y_true))

		torch.save(results,'predictions_{}_{}_{}_{}{}.pkl'.format('grid' if use_grid_model else 'image',"covar" if use_covariates else "nocovar",n_schools,effective_grid_len,'_residual' if train_residuals else ''))

		torch.save(dict(results=results,datasets=datasets),'results_saved_{}_{}_{}_{}{}.pkl'.format('grid' if use_grid_model else 'image',"covar" if use_covariates else "nocovar",n_schools,effective_grid_len,'_residual' if train_residuals else ''))

		results=pd.DataFrame(np.vstack((y_true,y_pred)).T,columns=['y_true','y_pred'])
		print(results)
		print('Test R2: {}, Test MAE: {}'.format(r2_score(results['y_true'].values,results['y_pred'].values), mean_absolute_error(results['y_true'].values,results['y_pred'].values)))
		results['County']=datasets['County']
		results['Population']=datasets['Population']
		results.to_csv('results_{}_{}_{}_{}{}.csv'.format('grid' if use_grid_model else 'image','covar' if use_covariates else "nocovar",n_schools,effective_grid_len,'_residual' if train_residuals else ''))

	else:
		dataloaders = {Set:DataLoader(image_datasets[Set],batch_size=32,num_workers=8,shuffle=False) for Set in ['train','val','test']}
		trainer.model.fc=trainer.model.fc[0][0]# remove [0]
		y_true=dataloaders[embedding_set].dataset.y#datasets.loc[datasets['Set']=='test']['Mortality'].values.flatten()
		images=dataloaders[embedding_set].dataset.images
		embeddings=trainer.predict(dataloaders[embedding_set])

		df=pd.DataFrame(embeddings,index=images)
		df['y_true']=y_true
		pickle.dump(df,open('{}_embeddings.pkl'.format(embedding_set),'wb'))


	"""
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, trim_mean
def output_r2_stats(t=0.3):
	d=load_data(t=t)
	return dict(test_r=pearsonr(d['test']['y_true'],d['test']['y_pred'])[0],
			test_r2=pearsonr(d['test']['y_true'],d['test']['y_pred'])[0]**2,
			val_r=pearsonr(d['val']['y_true'],d['val']['y_pred'])[0],
			val_r2=pearsonr(d['val']['y_true'],d['val']['y_pred'])[0]**2)

def output_mae_stats(t=0.3):
	d=load_data(t=t)
	return dict(test_mae=mean_absolute_error(d['test']['y_true'],d['test']['y_pred']),
			val_mae=mean_absolute_error(d['val']['y_true'],d['val']['y_pred']))

def load_data(t=0.3):
	d=torch.load("predictions_image.pkl")
	d['test']['y_true']=[np.mean(d['test']['y_true'][49*4*i:49*4*i+49*4]) for i in range(int(len(d['test']['y_true'])/(49*4)))]
	d['test']['y_pred']=[trim_mean(d['test']['y_pred'][49*4*i:49*4*i+49*4],t) for i in range(int(len(d['test']['y_pred'])/(49*4)))]
	d['val']['y_true']=[np.mean(d['val']['y_true'][49*4*i:49*4*i+49*4]) for i in range(int(len(d['val']['y_true'])/(49*4)))]
	d['val']['y_pred']=[trim_mean(d['val']['y_pred'][49*4*i:49*4*i+49*4],t) for i in range(int(len(d['val']['y_pred'])/(49*4)))]
	return d

def plot_data(t=0.3):
	d=load_data(t=t)
	plt.figure()
	plt.scatter(d['test']['y_pred'],d['test']['y_true'])
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.savefig('test_scatter.png')
	plt.figure()
	plt.scatter(d['val']['y_pred'],d['val']['y_true'])
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.savefig('val_scatter.png')

	"""


if __name__=='__main__':
	main()
