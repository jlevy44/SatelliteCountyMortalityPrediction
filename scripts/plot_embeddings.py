import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd, numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import dask.array as da
from os.path import join
import argparse
from skimage.io import imread
sns.set(style='white')
"""https://raw.githubusercontent.com/jlevy44/PathFlowAI/master/pathflowai/visualize.py"""

RANDOM_SEED=42
np.random.seed(RANDOM_SEED)

class PlotlyPlot:
	"""Creates plotly html plots."""
	def __init__(self):
		self.plots=[]

	def add_plot(self, t_data_df, G=None, color_col='color', name_col='name', xyz_cols=['x','y','z'], size=2, opacity=1.0, custom_colors=[]):
		"""Adds plotting data to be plotted.

		Parameters
		----------
		t_data_df:dataframe
			3-D transformed dataframe.
		G:nx.Graph
			Networkx graph.
		color_col:str
			Column to use to color points.
		name_col:str
			Column to use to name points.
		xyz_cols:list
			3 columns that denote x,y,z coords.
		size:int
			Marker size.
		opacity:float
			Marker opacity.
		custom_colors:list
			Custom colors to supply.
		"""
		plots = []
		x,y,z=tuple(xyz_cols)
		if t_data_df[color_col].dtype == np.float64:
			plots.append(
				go.Scatter3d(x=t_data_df[x], y=t_data_df[y],
							 z=t_data_df[z],
							 name='', mode='markers',
							 marker=dict(color=t_data_df[color_col], size=size, opacity=opacity, colorscale='Viridis',
							 colorbar=dict(title='Colorbar')), text=t_data_df[color_col] if name_col not in list(t_data_df) else t_data_df[name_col]))
		else:
			colors = t_data_df[color_col].unique()
			c = sns.color_palette('hls', len(colors))
			c = np.array(['rgb({})'.format(','.join(((np.array(c_i)*255).astype(int).astype(str).tolist()))) for c_i in c])#c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(colors) + 2)]
			if custom_colors:
				c = custom_colors
			color_dict = {name: c[i] for i,name in enumerate(sorted(colors))}

			for name,col in color_dict.items():
				plots.append(
					go.Scatter3d(x=t_data_df[x][t_data_df[color_col]==name], y=t_data_df[y][t_data_df[color_col]==name],
								 z=t_data_df[z][t_data_df[color_col]==name],
								 name=str(name), mode='markers',
								 marker=dict(color=col, size=size, opacity=opacity), text=t_data_df.index[t_data_df[color_col]==name] if 'name' not in list(t_data_df) else t_data_df[name_col][t_data_df[color_col]==name]))
		if G is not None:
			#pos = nx.spring_layout(G,dim=3,iterations=0,pos={i: tuple(t_data.loc[i,['x','y','z']]) for i in range(len(t_data))})
			Xed, Yed, Zed = [], [], []
			for edge in G.edges():
				if edge[0] in t_data_df.index.values and edge[1] in t_data_df.index.values:
					Xed += [t_data_df.loc[edge[0],x], t_data_df.loc[edge[1],x], None]
					Yed += [t_data_df.loc[edge[0],y], t_data_df.loc[edge[1],y], None]
					Zed += [t_data_df.loc[edge[0],z], t_data_df.loc[edge[1],z], None]
			plots.append(go.Scatter3d(x=Xed,
					  y=Yed,
					  z=Zed,
					  mode='lines',
					  line=go.scatter3d.Line(color='rgb(210,210,210)', width=2),
					  hoverinfo='none'
					  ))
		self.plots.extend(plots)

	def plot(self, output_fname, axes_off=False):
		"""Plot embedding of patches to html file.

		Parameters
		----------
		output_fname:str
			Output html file.
		axes_off:bool
			Remove axes.

		"""
		if axes_off:
			fig = go.Figure(data=self.plots,layout=go.Layout(scene=dict(xaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				yaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				zaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False))))
		else:
			fig = go.Figure(data=self.plots)
		py.plot(fig, filename=output_fname, auto_open=False)


def plot_embeddings(image_plot=True, embeddings_file='embeddings.pkl', fig_size=(30,30), outputfname='output_embedding.png', zoom=0.05, n_neighbors=10, sample_p=1., pca=False):
	""""""
	import dask
	from dask.distributed import Client
	from umap import UMAP
	import pickle
	import pandas as pd, numpy as np
	import skimage.io
	from skimage.transform import resize
	import matplotlib
	from sklearn.decomposition import PCA
	matplotlib.use('Agg')
	from matplotlib import pyplot as plt
	sns.set(style='white')

	def min_resize(img, size):
		"""
		Resize an image so that it is size along the minimum spatial dimension.
		"""
		w, h = map(float, img.shape[:2])
		if min([w, h]) != size:
			if w <= h:
				img = resize(img, (int(round((h/w)*size)), int(size)))
			else:
				img = resize(img, (int(size), int(round((w/h)*size))))
		return img

	#dask_arr = dask_arr_dict[ID]

	embeddings_df=pickle.load(open(embeddings_file,'rb'))

	if 0 and sample_p<1.:
		embeddings_df=embeddings_df.sample(frac=sample_p)

	umap=UMAP(n_components=2 if image_plot else 3,n_neighbors=n_neighbors, random_state=RANDOM_SEED) if not pca else PCA(n_components=2 if image_plot else 3, random_state=RANDOM_SEED)
	t_data=pd.DataFrame(umap.fit_transform(embeddings_df.iloc[:,:-1].values),columns=['x','y'] if image_plot else ['x','y','z'],index=embeddings_df.index)

	np.random.seed(RANDOM_SEED)

	if 0 and sample_p<1.:
		t_data=t_data.sample(frac=sample_p)

	t_data=t_data.sample(frac=sample_p)

	if image_plot:
		images=[]

		for img in t_data.index:#range(embeddings_df.shape[0]):
			images.append(dask.delayed(lambda x: imread(x))(img))#np.roll(Image.open(x),1,2)

		c=Client()
		images=dask.compute(*images)
		c.close()

		from matplotlib.offsetbox import OffsetImage, AnnotationBbox
		def imscatter(x, y, ax, imageData, zoom):
			images = []
			for i in range(len(x)):
				x0, y0 = x[i], y[i]
				img = imageData[i]
				#print(img.shape)
				image = OffsetImage(img, zoom=zoom)
				ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
				images.append(ax.add_artist(ab))

			ax.update_datalim(np.column_stack([x, y]))
			ax.autoscale()

		fig, ax = plt.subplots(figsize=fig_size)
		imscatter(t_data['x'].values, t_data['y'].values, imageData=images, ax=ax, zoom=zoom)
		sns.despine()
		plt.savefig(outputfname,dpi=300)
	else:
		t_data['name']=embeddings_df.index.values + np.vectorize(lambda x: ' Mortality={}'.format(round(x,2)))(embeddings_df.iloc[:,-1].values)

		t_data['color']=embeddings_df.iloc[:,-1]
		p=PlotlyPlot()
		p.add_plot(t_data)
		p.plot(outputfname,axes_off=True)

def main():
	p=argparse.ArgumentParser()
	p.add_argument('--image_plot', action='store_true')
	p.add_argument('--pca', action='store_true')
	p.add_argument('--embeddings_file',default='embeddings.pkl',type=str)
	p.add_argument('--sample_p',type=float,default=1.)
	p.add_argument('--fig_size',type=int,default=30)
	p.add_argument('--outputfname',type=str,default='output.embeddings.html')
	p.add_argument('--zoom',type=float,default=0.05)
	p.add_argument('--n_neighbors',type=int,default=10)

	kargs=vars(p.parse_args())
	kargs['fig_size']=(kargs['fig_size'],kargs['fig_size'])
	plot_embeddings(**kargs)

if __name__=='__main__':
	main()
