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
import math
from torch.autograd import Variable
import copy
from sklearn.metrics import r2_score, mean_absolute_error
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_model import *
import shap
import pickle
from PIL import Image

def main():
	p = argparse.ArgumentParser()
	p.add_argument('--model_loc', type=str) # ADD COUNTY
	p.add_argument('--county', type=str, default='best')
	p.add_argument('--batch_size', type=int, default=64)
	p.add_argument('--n_samples', type=int, default=1000)
	p.add_argument('--local_smoothing', type=float, default=0.8)
	p.add_argument('--dataset', type=str, default='data/training_datasets_original_new.csv')
	p.add_argument('--layer', type=str, default='')
	p.add_argument('--n_bootstrap', type=int, default=1)
	args=p.parse_args()
	grid_len=7
	n_schools=4
	multiplicative_factor=1000.
	dataset=args.dataset
	image_dir="image_data/"
	norm_file='norm.pkl'
	model_loc=args.model_loc
	method='gradient'
	county=args.county
	layer=args.layer
	num_targets=1
	outputfilename='shap.results.{}{}.png'.format(county,layer.replace(',','_'))
	local_smoothing=args.local_smoothing
	batch_size=args.batch_size
	n_samples=args.n_samples
	n_bootstrap=args.n_bootstrap
	n_outputs=1


	datasets_df=pd.read_csv(dataset,index_col=0)

	available_counties=[os.path.basename(d) for d in glob.glob(os.path.join(image_dir,'*')) if len(glob.glob(os.path.join(d,'*.png')))==n_schools*grid_len**2]

	datasets_df=datasets_df.reset_index(drop=True)
	datasets_df=datasets_df.loc[datasets_df['County'].isin(available_counties)]

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

	predictions=torch.load('predictions_{}.pkl'.format('image'))
	# dict(val=dict(y_pred=y_pred_val,y_true=y_true_val),test=dict(y_pred=y_pred,y_true=y_true)),

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
			dfff['Mortality_binned']=dff.iloc[i]['Mortality_binned']
			dfff['Set']=Set
			dfff['School']=school
			dfff['Idx']=image_number
			feature_extractor_data.append(dfff)

	feature_extractor_data=pd.concat(feature_extractor_data)
	print(feature_extractor_data)
	#feature_extractor_data=feature_extractor_data.sort_values(['Set','County','School','Idx'])
	feature_extractor_data_original=feature_extractor_data.copy()
	# if county in ['all','best']:
	np.random.seed(42)
	feature_extractor_data_representative=[]
	for Set, dff in feature_extractor_data.groupby('Set'):
		if Set in ['val','test']:
			if not (Set=='test' and county not in ['all','best']):
				dff['residual']=np.abs(dff['Mortality'].values-predictions[Set]['y_pred'])#[::-1]
				if Set=='test' and county=='all':
					feature_extractor_data_representative.append(dff.drop(columns=['Mortality_binned','residual']))
				else:
					for name, dfff in dff.groupby('Mortality_binned'):
						for County, dffff in dfff.groupby('County'):
							#print(county)
							dffff=dffff.iloc[np.argsort(dffff['residual'].values)[::-1][:2]]
							feature_extractor_data_representative.append(dffff.drop(columns=['Mortality_binned','residual']))
		else:
			feature_extractor_data_representative.append(dff)
	feature_extractor_data=pd.concat(feature_extractor_data_representative).sort_values(['Set','County','School','Idx'])

	if county not in ['all','best']:
		#feature_extractor_data,feature_extractor_data_test=feature_extractor_data[feature_extractor_data['Set']!='test'],feature_extractor_data[feature_extractor_data['Set']=='test']
		# feature_extractor_data=pd.concat([feature_extractor_data,feature_extractor_data_test[feature_extractor_data_test['County']==county]])
		feature_extractor_data=pd.concat([feature_extractor_data,feature_extractor_data_original[feature_extractor_data_original['County']==county]],join='inner')
	print(feature_extractor_data)

	transformers=generate_transformers(image_size=224, mean=[], std=[])

	norm_dataset=BasicImageSet(feature_extractor_data,'norm',transformers,add_idx=False)

	norm_opts=torch.load(norm_file)

	transformers=generate_transformers(image_size=224,**norm_opts)


	image_datasets={Set:BasicImageSet(feature_extractor_data,Set,transformers,add_idx=True) for Set in ['train','val','test']}
	torch.manual_seed(42)
	dataloaders = {Set:DataLoader(image_datasets[Set],batch_size=batch_size,num_workers=8,shuffle=True) for Set in ['train','val','test']} # (Set=='train')

	model=torch.load(model_loc)
	pretrained=True
	if model.name=='GridNetwork':
		if model.averaged_effect:
			model=nn.Sequential(model.feature_extraction,model.fc)
		else:
			print("Convolution attention temporarily not supported")
			exit()



	if torch.cuda.is_available():
		model=model.cuda()


	test_idx,X_test,y_test=next(iter(dataloaders['test']))
	test_idx=test_idx.detach().numpy().flatten()
	if torch.cuda.is_available():
		X_test=X_test.cuda()

	y_test=y_test.numpy()

	if y_test.shape[1]>1:
		y_test=y_test.argmax(axis=1)

	shap_all=[]

	if method=='gradient':
		if layer:
			try:
				layer_idx=list(map(int,layer.split(',')))
				layer=list(model.children())
				for idx in layer_idx:
					layer=layer[idx]
			except:
				layer=model._modules[layer]
			model2=(model,layer)
		else:
			model2=model

	for i in range(n_bootstrap):
		val_loader=iter(dataloaders['val'])
		_,background,y_background=next(val_loader)
		# if method=='gradient':
		# 	background=torch.cat([background,next(val_loader)[1]],0)
		if torch.cuda.is_available():
			background=background.cuda()

		if method=='deep':
			e = shap.DeepExplainer(model, background)
			s=e.shap_values(X_test)
		elif method=='gradient':
			e = shap.GradientExplainer(model2, background, batch_size=batch_size, local_smoothing=local_smoothing)
			s=e.shap_values(X_test, nsamples=n_samples, rseed=42)

		if n_outputs>1:
			shap_values, idx = s
		else:
			shap_values, idx = s, y_test

		shap_all.append(shap_values)

	#print(shap_values) # .detach().cpu()

	if num_targets == 1:
		shap_numpy = [sum([np.swapaxes(np.swapaxes(shap_values, 1, -1), 1, 2) for shap_values in shap_all])/len(shap_all)]
	else:
		print('Deprecated')
		exit()
	# else:
	# 	shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
		#print(shap_numpy.shape)

	y_pred = model(X_test).detach().cpu().numpy().flatten()
	no_google_label=False
	if no_google_label:
		X_test_numpy=X_test.detach().cpu().numpy()
		X_test_numpy=X_test_numpy.transpose((0,2,3,1))
		for i in range(X_test_numpy.shape[0]):
			X_test_numpy[i,...]*=np.array(norm_opts['std'])
			X_test_numpy[i,...]+=np.array(norm_opts['mean'])
		X_test_numpy=X_test_numpy.transpose((0,3,1,2))
		test_numpy = np.swapaxes(np.swapaxes(X_test_numpy, 1, -1), 1, 2)
	else:
		# NOT WORKING FOR NOW
		test_numpy=np.stack([np.array(Image.open(image_datasets['test'].images[i]).convert('RGB').resize((256,256))) for i in test_idx]).astype(float)
		def pad_with(vector, pad_width, iaxis, kwargs):
			pad_value = kwargs.get('padder', 0)
			vector[:pad_width[0]] = pad_value
			vector[-pad_width[1]:] = pad_value

		def pad_arr(new_width,data):
			old_width=data.shape[0]
			#print(data.shape)
			#print(old_width,new_width)
			arr=np.zeros((new_width,new_width,3))
			xi=int((new_width-old_width)/2)
			xf=new_width-xi
			#print(xi,xf)
			arr[xi:xf,xi:xf,:]=data
			return arr

		#print(shap_numpy[0].shape)
		shap_numpy=[np.array([pad_arr(256,shap_numpy[0][i]) for i in range(len(shap_numpy[0]))])] # range(test_numpy.shape[0]) np.stack np.pad(test_numpy[i],(256-244)/2,pad_with)
	if 1:
		labels=np.array(['True={},Predicted={}'.format(true,pred) for true,pred in zip(np.round(y_test.flatten(),3).astype(str).tolist(),np.round(y_pred,3).astype(str).tolist())]).reshape(-1,1)
		#labels==labels+y_pred.astype(str)
	else:
		labels = np.array([[(dataloader_val.dataset.targets[i[j]] if num_targets>1 else str(i)) for j in range(n_outputs)] for i in idx])#[:,np.newaxis] # y_test
	if 0 and (len(labels.shape)<2 or labels.shape[1]==1):
		labels=labels.flatten()#[:np.newaxis]
	#print(test_numpy)
	#print(labels.shape,shap_numpy.shape[0])
	print(shap_numpy, test_numpy, labels)
	plt.figure()
	shap.image_plot(shap_numpy, test_numpy, labels)# if num_targets!=1 else shap_values -test_numpy , labels=dataloader_test.dataset.targets)
	plt.savefig(outputfilename, dpi=300)




if __name__=='__main__':
	main()
