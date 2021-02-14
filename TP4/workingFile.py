import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import ndimage

import maxflow

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from skimage.io import imread,imsave

import skimage.filters
from scipy.cluster.vq import kmeans2

plt.ion()

def computeNormalizedImage(im,display_distribution=False):
	# TODO compute the normalized color image by dividing each channel by the norm of the rgb vector for that pixel

	im_norm = np.linalg.norm(im, axis=-1)[..., np.newaxis]
	im_intensity_normalised = im / im_norm

	if display_distribution:
		#display the disribution of normalized color, should be on the unity sphere
		points=im_intensity_normalised.reshape(-1,3)
		fig=plt.figure(figsize=(8,8))
		ax = fig.add_subplot(111, projection='3d')
		colormap=np.array([[0,0,200],[250,250,0],[0,200,0],[200,0,0]],dtype=np.uint8)
		# ax.scatter(points[::100,0],points[::100,1],points[::100,2],c=points[::100,:],edgecolors='none')
		ax.set_xlabel('red')
		ax.set_ylabel('green')
		ax.set_zlabel('blue')
		ax.set_xlim([0,1])
		ax.set_ylim([0,1])
		ax.set_zlim([0,1])
		plt.draw()
		plt.show()

	return im_intensity_normalised

def displayLabelsImage(imlabels):
	# Displaying the labels image
	colormap=np.array([[250,250,0],[0,0,200],[0,200,0],[200,0,0]],dtype=np.uint8)
	imlabelsColors=colormap[imlabels,:]
	plt.figure()
	plt.imshow(imlabelsColors)
	plt.show()


# TODO
# Define your own kmeans function


def imageKmeans(im_intensity_normalised,nb_clusters):
	# TODO
	# from the image, create a vector of point colors of size N by 3 with N the number of pixels in the image
	# and you can call the function kmeans2 from scipy.cluster.vq
	# before defining your own kmeans function,
	# retrieve both the centers and the labels from this function
	# reshape the labels to have the size of the image to get an image called imlabels
	points = im_intensity_normalised.reshape(-1, 3)
	nb_iter = 10
	centroids, labels = scipy.cluster.vq.kmeans2(points, k=nb_clusters, iter=nb_iter, minit='points')
	point_labels = np.array([labels[point_idx] for point_idx in range(points.shape[0])])
	img_labels = np.reshape(point_labels, im_intensity_normalised.shape[:2])

	return img_labels, centroids

def displayUnaryLabelCosts(costs):
	plt.figure()
	# print("Costs.shape[2]:", Costs.shape[2])
	# TODO : improve this part to be more general
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.imshow(costs[:,:,i],cmap=plt.cm.Greys_r)

def unaryLabelCosts(im_intensity_normalized,mean_colors):
	# TODO
	# given a matrix of color of size N by 3
	# Compute an array Costs of size H by H by nb_clusters
	# such that Costs[i,j,k]= norm(im[i,j,:]-colors[:,k])**2
	# you can avoid loops by using numpy broadcasting
	# http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

	# Solution without broadcasting:

	# Solution using broadcasting:
	# we create first an array of size H x W x 1 x 3 from an array of size H x W x 3 using [:,:,None,:] to create a new axis of size one
	# then we can substract an array of size K x 3 from an array of size H x W x 1 x 3 using broadcasting
	# to get an array of size H x W x K x 3 and we sum the square along the last dimension
	costs = np.linalg.norm(im_intensity_normalized[:, :, np.newaxis, :] - mean_colors, axis=-1) ** 2
	return costs


def graphCutSegment(costs_class0,costs_class1,imlabelsBinary):

	H=costs_class0.shape[0]
	W=costs_class0.shape[1]
	nbpixels=H*W

	# -----------------------Segmentation method using regional data term and constant edge cost-------------------------------

	g = maxflow.Graph[float](nbpixels,nbpixels * 2)


	# TODO 
	# encode unary term of the energy shown in those slides of graph-cut (i.e., sum_p Dp(lp) +sum_pq Vqp(lq,lp))
	# by adding arcs in the graph by calling the function g.add_tweights_vectorized 
	# this function allows to add arcs between nodes and the source node s and a the sink node t (oriented as source->node and node->target)
	# g.add_tweights_vectorized(indices_nodes,costs_arcs_from_s_to_nodes,costs_arcs_from_nodes_to_t) 
	# with indices_nodes,costs_arcs_from_s_to_nodes,costs_arcs_from_nodes_to_t vector whose lenght is the number of pixels
	# WARNING, in the contrary to the slide  24 , the pymaxflow code seems to use the convention 
	# that s is in the set of node with labels 0 and t is in the set of node with labels 1
	# look at the graph slide 24 to see how to go from Dp(0) and Dp(1) to the arc weights keeping 
	# in mind that pymaxflow use a opposite convention
	# make sure that indices_nodes,costs_arcs_to_s and costs_arcs_to_t are vectors of the same size
	# otherwise will get an assertion error 
	# Note :the unvectorized version is documented in the graph.h file un pymaxflow

	indice_nodes = g.add_node(nbpixels)
	costs_arcs_from_s_to_nodes = costs_class0
	costs_arcs_from_nodes_to_t = costs_class1
	g.add_tweights_vectorized(
		indice_nodes.astype(np.int32),
		costs_arcs_from_s_to_nodes.astype(np.float32),
		costs_arcs_from_nodes_to_t.astype(np.float32))


	#We check that the minimization of this energy without extra node gives back the same labels as imlabelsBinary
	print("calling maxflow")
	g.maxflow()
	out = g.what_segment_vectorized()    
	imlabels2=out.reshape((H,W))
	assert(np.all(imlabelsBinary==imlabels2))

	# TODO
	# encode binary terms of the energy in slide 25 (sum_p Dp(fp) +sum_pq Vqp(fq,fp))
	# by considering first horizontal edges between neighboring pixels then vertical edges
	# you  add oriented arcs in the graph between nodes by calling the function g.add_edge_vectorized
	# g.add_edge_vectorized(indices_node_i,indices_node_j,weight_arc_i_to_j,weight_arc_j_to_i)
	# in our case we use weight_arc_i_to_j=weight_arc_i_to_j=alpha
	# first create a matrix indices = np.arange(nbpixels) .reshape(H,W) an use 
	# then use submaxtrices extracted from  indices then flattened to create indices_node_i and indices_node_j
	# Note : you can have a look on the example in test.py in the pymaxflow directory

	alpha=0.1


	indices = np.arange(nbpixels) .reshape(H,W).astype(np.int32)
	#------ adding horizontal edges------
	indices_node_i = indices[:, :-1].ravel()
	indices_node_j = indices[:, 1:].ravel()
	cost_diff = alpha
	g.add_edge_vectorized(indices_node_i, indices_node_j, cost_diff ,cost_diff)  


	#------ adding vertical edges------
	indices_node_i = indices[1:, :-1].ravel()
	indices_node_j = indices[:-1, 1:].ravel()
	# cost_diff=n ?
	g.add_edge_vectorized(indices_node_i, indices_node_j, cost_diff, cost_diff)

	indices_node_i = indices[:-1, 1:].ravel()
	indices_node_j = indices[1:, :-1].ravel()
	g.add_edge_vectorized(indices_node_i, indices_node_j, cost_diff, cost_diff)

	# Getting the result image
	print("calling maxflow")
	g.maxflow()
	out = g.what_segment_vectorized()
	

	imlabelsGC=out.reshape((H,W)) 
	return imlabelsGC

def main():
	im=np.array(imread('tiger.png')).astype(np.float)
	H=im.shape[0]
	W=im.shape[1]
	nbpixels=H*W   

	im_intensity_normalized=computeNormalizedImage(im,display_distribution=True)

	nb_clusters=4
	random.seed((1000,2000))# make sure the kmean call is repeatable to check correctness of te code
	imlabels,mean_colors=imageKmeans(im_intensity_normalized,nb_clusters =nb_clusters)

	displayLabelsImage(imlabels) 


	Costs=unaryLabelCosts(im_intensity_normalized,mean_colors)
	displayUnaryLabelCosts(Costs)



	# retrive the class that is the most   represented in the center region of the image    
	classSegment=np.argmax(np.bincount(imlabels[100:150,150:250].flat)) 

	# New cost using only two classes by grouping all other classes into a single class ,
	# we take the minium of the cost in the set of other classes to get the unary cost for new 
	# aggregated class   
	otherClasses=[i for i in range(nb_clusters) if i!=classSegment ]  
	costs_class0=Costs[:,:,classSegment]
	costs_class1=np.min(Costs[:,:,otherClasses],axis=2)
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(costs_class0,cmap=plt.cm.Greys_r)
	plt.subplot(1,2,2)    
	plt.imshow(costs_class1,cmap=plt.cm.Greys_r)    

	imlabelsBinary=imlabels!=classSegment

	imlabelsGC=graphCutSegment(costs_class0,costs_class1,imlabelsBinary)

	plt.ioff()
	
	displayLabelsImage(imlabelsGC)


if __name__ == "__main__":
	main()
