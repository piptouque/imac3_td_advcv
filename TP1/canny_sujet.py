#!/usr/bin/python
# -*- coding: utf-8 -*-
# Attentions : faire adjustments necessaires pour les differentes versions de Python



import numpy as np
import numpy.linalg as lin
import scipy.ndimage
import imageio
import pickle
import matplotlib.pyplot as plt
# canny

def pause():
    """this function allows to refresh a figure and do a pause"""
    plt.draw() 
    plt.pause(0.001)
    
def plotImageWithColorBar(I,cmap=plt.cm.Greys_r,title=''): 
    """this function displays an image and a color bar"""
    fig = plt.figure(figsize=(8,7))
    ax=plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0)
    implot=plt.imshow(I,cmap)
    pause()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(implot, use_gridspec=True)    
    plt.title(title)
    
    

def displayImageAndVectorField(I,vx,vy,step,color):
    """this function displas an image with an overlaid vector field"""
    assert (vx.shape==vy.shape)
    assert (I.shape==vy.shape)
    fig = plt.figure(figsize=(8,7))
    ax=plt.subplot(1,1,1)
    plt.imshow(I,cmap=plt.cm.Greys_r)
    pause()
    X,Y = np.meshgrid( np.arange(0,I.shape[1]),np.arange(0,I.shape[0]) )
    plt.quiver(X[::step,::step],Y[::step,::step],vx[::step,::step],-vy[::step,::step],color=color)
    # dans le cas où l'on affiche une image on a besoin d'inverser la direction y , étrange ...
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0)
    
    
def sampleImage(I,x,y):
    """cette fonction permet de sampler une image en niveau de gris 
    en un ensemble de positions discrètes fournies dans deux matrices 
    de même dimension mxn le resultat est de taille mxn
    samples[i,j]=I[y[i,j],x[i,j]]"""
    assert np.all(x.shape == y.shape)
    x=x.astype(int)
    y=y.astype(int) 
    return I[y,x] # see http://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays

def smoothGaussian(im,sigma):
    # implementez un  lissage de l'image en utilisant un noyau gaussien avec un déviation standard de sigma
    # prenez un taille de noyeau de 3*sigma de chaque coté du centre 
    # n'oubliez de convertir l'image en float avant de faire la convolution avec im=im.astype(float)
    kernel_base_size = 3
    # n'utilisez pas de fonction toute faite pour obtenir le noyau gaussien ,
    # vous pouvez utiliser np.meshgrid (utilisée dans la fonction displayImageAndVectorField ci dessus) avec la fonction np.arange
    kernel_size = kernel_base_size * 2 + 1
    mu = 0
    xx, yy = np.meshgrid(
        np.arange(- kernel_size / 2, kernel_size / 2 + 1),
        np.arange(- kernel_size / 2, kernel_size / 2 + 1))
    kernel = np.exp(- (((xx**2 + yy**2).astype(float) - mu) / sigma) ** 2)
    # attention np.arange(a,b) donne des nombres allant de a à b-1 et non de a à b
    # pour créer deux tableau 2D carrées X et Y de taille 2N+1 avec X[i,j]=j-N et Y[i,j]=i-N avec N=*3*sigma
    # puis utilisez les operateur **2 et np.exp sur ces tableau pour obtenir une image de gaussienne centree en (N,N)
    # affichez cette image avec plt.imshow(gaussienne)
    plt.imshow(kernel)
    pause()
    # si l'image semble quantifiée  c'est parceque vous avez oublié de convertir X et Y en float avec X=X.astype(float) et Y=Y.astype(float)
    # avant de faire une division
    # verifiez numériquement la symétrie de votre gaussienne
    print("Erreur de symétrie: ", (lin.norm(np.transpose(kernel) - kernel)))
    # vous pouvez implémenter une autre version qui tire avantage de la séparabilité du filtre
    im_smooth = im.astype(float)
    im_smooth = scipy.ndimage.convolve(im_smooth, kernel, mode='constant')
    plt.imshow(im_smooth)
    pause()

    return im_smooth
    
def gradient(im_smooth):
    # utilisez scipy.ndimage.convolve pour calculer le gradient de l'image
    # assurez vous d'avoir péalablement converti l'image en float sans quoi vous n'aurez pas le résultat souhaité
    # verifiez que vous avez n'avez pas calculé le vecteur inverse du gradient i.e que le flèches pointent bien vers les zones plus claires
    # lorsque vous executez displayImageAndVectorField(im_smooth,gradient_x,gradient_y,10,'b')  
    # ...
    kernel_x = np.array([[1., -1], [0., 0.]])
    kernel_y = np.array([[-1., 0.], [1., 0.]])

    gradient_x = scipy.ndimage.convolve(im_smooth, kernel_x, mode='constant')
    gradient_y = scipy.ndimage.convolve(im_smooth, kernel_y, mode='constant')
    displayImageAndVectorField(im_smooth, gradient_x, gradient_y, 10, 'b')
    pause()
    return gradient_x,gradient_y


def gradientNormeEtAngle(gradient_x,gradient_y):
    # calculez la norm du gradient pour chaque pixel
    # et utilisez la fonction np.arctan2 pour calculer l'angle du gradient pour chaque pixel, attention l'ordre des deux argument est important
    # ...
    norm_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    angle = np.arctan2(gradient_y, gradient_x)
    return norm_gradient, angle

def approxAngleEtDirection(angle): 
    # arrondissez chaque angle donné à l'angle multiple de pi/4 le plus proche (multipliez par 4/pi , arrondissez , puis remultipliez par pi/4)
    # puis arrondissez pour chaque pixel le vecteur unitaire [cos(angle_rounded),sin(angle_rounded)] 
    # au vecteur à coordonnée entières le plus proche i.e. dans [1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]
    # pour obtenir deux images direction_x et direction_y
    # avec [direction_x[i,j],direction_y[i,j]] le vecteur à coordonnées entière le plus proche de [cos(angle_rounded[i,j]),sin(angle_rounded[i,j])] 
    # essayez d'utiliser le caclul matriciel pour éviter de faire des boucles sur les pixels
    # Completez ici  
    angle_rounded = np.round(angle * (4 * np.pi)) * 4 * np.pi
    direction_x = np.round(np.cos(angle_rounded))
    direction_y = np.round(np.sin(angle_rounded))
    # ces lignes permettent d'éviter de sortir de l'image dans la fonction localMaximum
    direction_x[:,0]=0 
    direction_x[:,-1]=0
    direction_y[0,:]=0
    direction_y[-1,:]=0    
    return angle_rounded, direction_x, direction_y
    
def localMaximum(norm_gradient,direction_x,direction_y):
    # ecrivez cette fonction qui renvoi une image binaire avec maxi[i,j]==True si le pixel (i,j) et un
    # maximum local dans la direction donnée par direction _x et direction_y
    # i.e si norm_gradient[i,j]>=norm_gradient[i+direction_y[i,j],j+direction_x[i,j]]
    #    et  norm_gradient[i,j]>=norm_gradient[i-direction_y[i,j],j-direction_x[i,j]] 
    # essayer de vectorizer cette fonction pour éviter de faire une boucle sur les pixels
    # en utilisant X,Y,direction_x , direction_y et  la fonction sampleImage définie plus haut 
    
    X,Y = np.meshgrid( np.arange(0,norm_gradient.shape[1]),np.arange(0,norm_gradient.shape[0]) )
    a = sampleImage(norm_gradient,X,Y) #samples au centre
    right = sampleImage(norm_gradient, X + direction_x, Y + direction_y)
    left = sampleImage(norm_gradient, X - direction_x, Y - direction_y)
    #...
    maxi = np.logical_and(np.greater(a, right), np.greater(a, left))
    return maxi
 
def hysteresis(maxi, norm_gradient, seuil1, seuil2) :
    # TODO: codez l'hysteresis (voir le cours)
    assert(seuil1 <= seuil2)
    # 1) commencer par obtenir m1 et m2
    m_maxi = np.where(maxi, norm_gradient, 0.)
    m1 = np.where(m_maxi >= seuil1, 1, 0)
    m2 = np.where(m_maxi >= seuil2, 1, 0)

    # 2) puis obtenez une image qui vaut 0 en dehors des bords et une valeur entière
    #    qui est constant sur chaque courbe connectée de m1 avec la fonction ndimage.label 
    #    (Attention: utilisez un voisinage 8 neighbourhood pour la fonction label...)
    #    ndimage.label renvoie un tuple dont le premier élement est un tableau dans lequels
    #    les 1 du tableau donné en entré sont remplacés par un entier qui correspond
    #    au numéro de la courbe connectée
    weak_adj_structure = np.ones((3, 3))
    m_curves, number_curves = scipy.ndimage.label(m1, weak_adj_structure)

    # 3) utilisez la fonction ndimage.maximum  pour obtenir un vecteur dont la longeur 
    #    est le nombre de courbes connectées et dont le ieme element contient le maximum
    #    que prend l'image m2 le long de la ieme courbe obtenue. le ieme element de ce 
    #    vecteur sera donc 
    #      - 1 s'il existe un pixel de la ieme courbe de m1 pour lequel m2
    #        est 1 (c'est a dire dont le gradient est supérieur à seuil2)  
    #      - 0 sinon
    curve_maxima = scipy.ndimage.maximum(m2, labels=m_curves, index=np.arange(number_curves + 1))


    # 4) creez l'image edges en le fait que A[B] avec A un vecteur et B une matrice 
    #    à coefficient entiers donne une matrice C de même taille que B avec C[i,j]=A[B[i,j]] 
    #   (see http://docs.scipy.org/doc/numpy/user/basics.indexing.html#index-arrays in case 
    #    the index array is multidimensional)
    print(number_curves)
    print(len(curve_maxima))
    edges = curve_maxima[m_curves]
    print(norm_gradient.shape)
    print(edges.shape)
    plt.imshow(edges)
    pause()
    return m1, m2, edges
    
def canny(im,sigma,seuil1,seuil2,display=True):
    
    im_smooth=smoothGaussian(im,sigma)
    
    gradient_x,gradient_y=gradient(im_smooth)
    
    norm_gradient,angle=gradientNormeEtAngle(gradient_x,gradient_y)
    
    angle_rounded,direction_x,direction_y=approxAngleEtDirection(angle)
    
    maxi=localMaximum(norm_gradient,direction_x,direction_y)
    
    m1,m2,edges=hysteresis(maxi,norm_gradient,seuil1,seuil2)
    
    
    with open('canny_etapes.pkl', 'wb') as f:
        l = [im_smooth.astype(np.float16),
           gradient_x.astype(np.float16),
           gradient_y.astype(np.float16),
           norm_gradient.astype(np.float16),
           angle.astype(np.float16),
           angle_rounded.astype(np.float16),
           direction_x.astype(np.int8),
           direction_y.astype(np.int8),
           maxi, m1, m2, edges]
        pickle.dump(l,f)

    with open('canny_etapes.pkl', 'rb') as f:
        im_smooth,gradient_x,gradient_y,norm_gradient,angle,\
        angle_rounded,direction_x,direction_y,maxi,m1,m2,edges = pickle.load(f)


    if display:
        plt.ion()
        plotImageWithColorBar(im_smooth,title='im_smooth')
        pause()
        plotImageWithColorBar(gradient_x,title='gradient x')
        pause()
        plotImageWithColorBar(gradient_y,title='gradient x')
        pause()
        displayImageAndVectorField(im_smooth,gradient_x,gradient_y,10,'b')  
        pause()
        plotImageWithColorBar(norm_gradient,title='norm_gradient')
        pause()
        plotImageWithColorBar(angle,cmap=plt.cm.hsv,title='angle') 
        pause()
        plotImageWithColorBar(angle_rounded,cmap=plt.cm.hsv)
        pause()
        displayImageAndVectorField(im_smooth,direction_x,direction_y,15,'b')
        pause()
        plotImageWithColorBar(m1)
        pause()
        plotImageWithColorBar(m2)   
        pause()
        plotImageWithColorBar(edges) 
        pause()

    return edges

def main():
    im = imageio.imread('einstein.jpg')
    sigma=2
    seuil1 = 30
    seuil2 = 200
    # La partie d'affichage de canny() ne fonctionne pas sur ma machine.
    display = False
    edges = canny(im, sigma, seuil1=seuil1, seuil2=seuil2, display=display)

if __name__ == "__main__":
    main()
