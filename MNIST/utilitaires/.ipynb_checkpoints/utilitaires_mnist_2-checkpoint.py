# coding=utf-8

###### VERSION 2/7 #####

# Import des librairies utilisées dans le notebook
import requests
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from zipfile import ZipFile
from io import BytesIO, StringIO
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
#from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd

from utilitaires_common import *

plt.rcParams['figure.dpi'] = 200

from IPython.display import display # Pour afficher des DataFrames avec display(df)

# Pour afficher les dataframe pandas en HTML
from IPython.display import display_html
from itertools import chain, cycle

### --- IMPORT DES DONNÉES ---
# Téléchargement et extraction des inputs contenus dans l'archive zip

print("Chargement de la base de donnée d'images en cours...")

inputs_zip_url = "https://raw.githubusercontent.com/akimx98/challenge_data/main/input_mnist_2.zip"
inputs_zip = requests.get(inputs_zip_url)
zf = ZipFile(BytesIO(inputs_zip.content))
zf.extractall()
zf.close()


# Téléchargement des outputs d'entraînement de MNIST-2 contenus dans le fichier y_train_2.csv
output_train_url = "https://raw.githubusercontent.com/akimx98/challenge_data/main/y_train_2.csv"
output_train = requests.get(output_train_url)

output_train_chiffres_url = "https://raw.githubusercontent.com/akimx98/challenge_data/main/y_train_2_chiffres.csv"
output_train_chiffres = requests.get(output_train_chiffres_url)


# MNIST-2
chiffre_1 = 2
chiffre_2 = 7
chiffres = [chiffre_1, chiffre_2]
classes = [-1,1]

# # Inputs
with open('mnist_2_x_train.pickle', 'rb') as f:
    ID_train_2, d_train_2 = pickle.load(f).values()

with open('mnist_2_x_test.pickle', 'rb') as f:
    ID_test_2, d_test_2 = pickle.load(f).values()

# Outputs
_, r_train = [np.loadtxt(StringIO(output_train.content.decode('utf-8')),
                                dtype=int, delimiter=',')[:,k] for k in [0,1]]

_, r_train_chiffres = [np.loadtxt(StringIO(output_train_chiffres.content.decode('utf-8')),
                                dtype=int, delimiter=',')[:,k] for k in [0,1]]

# Ici le d_train c'est celui de MNIST-2
d_train = d_train_2
d_test = d_test_2

# VERSION 2/7 : 
r_train = r_train_chiffres
classes = chiffres

N = len(d_train)

d_train_par_population = [d_train[r_train==k] for k in classes]

d = d_train[10,:,:].copy()

print("Images chargées !")

# Pour que ce soit défini une fois, au cas où l'élève n'excéuter pas la cellule def caractéristique
y_petite_caracteristique = 7
y_grande_caracteristique = 2

def caracteristique(d):
    """Fonction qui calcule la caractéristique d'une image d"""
    # Nous avons codé la fonction moyenne qui prend une image en paramètre
    k = moyenne(d)
    return k

# Fonction répondant au problème en fonction de la caractéristique k de l'image que l'on doit classer
def classification(k, x):
    global y_petite_caracteristique, y_grande_caracteristique
    """Fonction qui répond à la question : est-ce un 2 ou un 7 ?"""
    # Comparaison de la caractéristique au seuil t
    if k < x:
        return y_petite_caracteristique
    else:
        return y_grande_caracteristique

def calcul_caracteristiques(d_train, caracteristique):
    """Fonction qui calcule les caractéristiques de toutes les images de d_train"""
    vec_caracteristique = np.vectorize(caracteristique, signature="(m,n)->()")
    return vec_caracteristique(d_train)

# Calculer l'estimation et l'erreur : 
def erreur_train(d_train, r_train, x, classification, caracteristique):
    """Fonction qui calcule l'erreur d'entraînement pour un seuil t donné"""
    return erreur_train_optim(calcul_caracteristiques(d_train, caracteristique),r_train,x,classification)

# Calculer l'estimation et l'erreur a partir du tableau de caractéristique des images : 
def erreur_train_optim(k_d_train, r_train, x, classification):
    # Vectorize the classification function if it's not already vectorized
    r_train_est = np.vectorize(classification)(k_d_train,x)
    
    # Calculate the mean error by comparing the estimated y values with the actual r_train values
    return np.mean(r_train_est != r_train)
        

# Erreurs
# Fonction qui calcule l'erreur pour une image d, en fonction de la réponse y et du paramètre s
def erreur_image(d, y, s):
    global caracteristique, classification, N
    c = caracteristique(d)
    y_est = classification(d, s)
    
    if y_est == y:
        erreur = 0
    else:
        erreur = 1
        
    return erreur

# Fonction qui calcule la moyenne des erreur par image pour donner l'erreur d'entrainement
def erreur_train_bis(d_train, r_train, s):
    liste_erreurs = []

    for i in range(N):
        d = d_train[i]
        r = r_train[i]
        
        erreur = erreur_image(d, r, s)
        liste_erreurs.append(erreur)
        
    return moyenne(liste_erreurs)

# def erreur_lineaire(m, p,c_train):
#     liste_erreurs = []

#     # On remplit y_est_train à l'aide d'une boucle :
#     for i in range(N):
#         x = x_train[i]
#         y = r_train[i]
#         x_1, x_2 = c_train[i]
#         y_est = classificateur(m, p, x_1, x_2)

#         if y_est == y:
#             erreur = 0
#         else:
#             erreur = 1

#         liste_erreurs.append(erreur)

#     return np.mean(liste_erreurs)


def titre_image(rng):
    titre = "y = "+str(r_train[rng])+" (chiffre = "+str(r_train_chiffres[rng])+")"
    return titre

def imshow(ax, image, **kwargs):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, extent=[0, 28, 28, 0], **kwargs)

def outline_selected(ax, a=None, b=None, displayPoints=False, zoneName=None, zoneNamePos='right', nameA='A', nameB='B', color='red'):
    if a is not None:
        if b is None:
            b = a
            
        numero_ligne_debut = min(a[0], b[0])
        numero_ligne_fin = max(a[0], b[0])
        numero_colonne_debut = min(a[1], b[1])
        numero_colonne_fin = max(a[1], b[1])
    
        if numero_ligne_debut < 0 or numero_colonne_debut < 0 or numero_ligne_fin > 27 or numero_colonne_fin > 27:
            print_error("Les valeurs des index doivent être compris entre 0 et 27.")
            return

        padding = 0.2  # adjust this value as needed
        if a == b:
            padding = -0.1
        rect = mpatches.Rectangle((numero_colonne_debut + padding, numero_ligne_debut + padding), 
                                 numero_colonne_fin - numero_colonne_debut + 1 - 2 * padding, 
                                 numero_ligne_fin - numero_ligne_debut + 1 - 2 * padding, 
                                 fill=False, edgecolor=color, lw=2)
        ax.add_patch(rect)

        if displayPoints and a != b:
            if a[1] <= b[1]:
                ha_a = 'right'
                ha_b = 'left'
            else:
                ha_a = 'left'
                ha_b = 'right'

            if a[0] <= b[0]:
                va_a = 'bottom'
                va_b = 'top'
            else:
                va_a = 'top'
                va_b = 'bottom'

            ax.text(a[1], a[0], nameA, ha=ha_a, va=va_a, color=color, fontsize=12, fontweight='bold')
            ax.text(b[1], b[0], nameB, ha=ha_b, va=va_b, color=color, fontsize=12, fontweight='bold')

        if zoneName is not None:
            if zoneNamePos == 'right':
                col = 32
                ha = 'left'
            elif zoneNamePos == 'left':
                col = -6
                ha = 'right'
            elif zoneNamePos == 'center':
                col = (numero_colonne_debut + numero_colonne_fin) / 2
                ha = 'center'
            else:
                raise ValueError("zoneNamePos doit valoir 'right', 'left' ou 'center'")

            ax.text(col,
                    (numero_ligne_debut + numero_ligne_fin) / 2,
                    zoneName, ha=ha, va='center', color='red', fontsize=12, fontweight='bold')


# Affichage d'une image
def affichage(image, a=None, b=None, displayPoints=False, titre=""):
    """Fonction qui affiche une image avec un rectangle rouge délimité par les points a et b
        a : tuple (ligne, colonne) représentant le coin en haut à gauche du rectangle
        b : tuple (ligne, colonne) représentant le coin en bas à droite du rectangle
    Si displayPoints est True, les points A et B sont affichés
    """
    if image.min().min() < 0 or image.max().max() > 255:
        print_error("fonction affichage : Les valeurs des pixels de l'image doivent être compris entre 0 et 255.")
        return

    fig, ax = plt.subplots(figsize=(3,3))
    imshow(ax, image)
    ax.set_title(titre)
    ax.set_xticks(np.arange(0,28,5))
    ax.xaxis.tick_top()
    ax.set_title(titre)
    ax.xaxis.set_label_position('top') 
    outline_selected(ax, a, b, displayPoints)

    plt.show()
    plt.close()

def affichage_2(image1, image2, A=None, B=None, displayPoints=False, titre1="", titre2="", histogramme=False):
    """Fonction qui affiche deux images côte à côté, avec un rectangle rouge délimité par les points A et B
        A : tuple (ligne, colonne) représentant le coin en haut à gauche du rectangle
        B : tuple (ligne, colonne) représentant le coin en bas à droite du rectangle
    Si displayPoints est True, les points A et B sont affichés
    """
    if image1.min().min() < 0 or image1.max().max() > 255:
        print_error("fonction affichage : Les valeurs des pixels des images doivent être compris entre 0 et 255.")
        return
    
    if image2.min().min() < 0 or image2.max().max() > 255:
        print_error("fonction affichage : Les valeurs des pixels des images doivent être compris entre 0 et 255.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    imshow(ax[0], image1)
    ax[0].set_title(titre1)
    ax[0].set_xticks(np.arange(0,28,5))
    ax[0].xaxis.tick_top()
    outline_selected(ax[0], A, B, displayPoints)

    imshow(ax[1], image2)
    ax[1].set_title(titre2)
    ax[1].set_xticks(np.arange(0,28,5))
    ax[1].xaxis.tick_top()
    outline_selected(ax[1], A, B, displayPoints)

    plt.tight_layout()

    plt.show()
    plt.close()
    
    if histogramme:
        affichage_dix_caracteristique(a=A, b=B, avec_df=False)
        caracteristique = get_variable('caracteristique')
        afficher_histogramme(caracteristique)
    else:
        affichage_dix_caracteristique(a=A, b=B)
    
    return

# Affichage d'une image sous forme de tableau
pd.set_option('display.max_rows', 28)
pd.set_option('display.max_columns', 28)
# Set the width of each column
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('colheader_justify', 'center')

def affichage_tableau(image, a=None, b=None):
    """Fonction qui affiche une image sous forme de tableau avec un rectangle rouge délimité par les points a et b
        a : tuple (ligne, colonne) représentant le coin en haut à gauche du rectangle
        b : tuple (ligne, colonne) représentant le coin en bas à droite du rectangle
        Si a et b ne sont pas fournis, la fonction affiche l'image sans rectangle rouge."""
    df = pd.DataFrame(image)
    if a is not None:
        if b is not None:
            numero_ligne_debut = min(a[0], b[0])
            numero_ligne_fin = max(a[0], b[0])
            numero_colonne_debut = min(a[1], b[1])
            numero_colonne_fin = max(a[1], b[1])
        else:
            numero_ligne_debut = a[0]
            numero_ligne_fin = a[0]
            numero_colonne_debut = a[1]
            numero_colonne_fin = a[1]
    
        if numero_ligne_debut < 0 or numero_colonne_debut < 0 or numero_ligne_fin > 27 or numero_colonne_fin > 27:
            print_error("Les valeurs des index doivent être compris entre 0 et 27.")
            return

        slice_ = (slice(numero_ligne_debut, numero_ligne_fin), slice(numero_colonne_debut, numero_colonne_fin))
        try:
            s = df.style.set_properties(**{'background-color': 'red'}, subset=slice_)
            display(s)
            return
        except:
            if b is not None:
                # return df.iloc[max(0, numero_ligne_debut - 1):min(len(image), numero_ligne_fin+2), max(0, numero_colonne_debut - 1):min(len(image), numero_colonne_fin+2)]
                display(df.iloc[numero_ligne_debut:numero_ligne_fin+1, numero_colonne_debut:numero_colonne_fin+1])
                return 
    display(df)
    return

def affichage_dix_caracteristique(predictions=False, a=None, b=None, avec_df = True):
    affichage_dix(d_train, a, b)
    if avec_df:
        df = pd.DataFrame()
        df['$r$ (classe)'] = r_train[0:10]   
        caracteristique = get_variable('caracteristique')
        df['$k$ (caracteristique)'] = [caracteristique(d) for d in d_train[0:10]]
        if predictions:
            df['$\hat{r}$ (prediction)'] = '?'
        df.index+=1

        display_html(df.to_html(index=False), raw=True)
    return

# Affichage 10 avec les valeurs de y en dessous
def affichage_dix(images=d_train, a=None, b=None, zones=[], liste_y = r_train, n=10, axes=False):
    global r_train
    fig, ax = plt.subplots(1, n, figsize=(n, 1))
    
    # Cachez les axes des subplots
    for j in range(n):
        if axes:
            ax[j].set_xticks(np.arange(0,28,5))
            ax[j].set_yticks(np.arange(0,28,5))
            ax[j].xaxis.tick_top()
            ax[j].tick_params(axis='both', which='major', labelsize=7)

        else:
            ax[j].axis('off')
        imshow(ax[j], images[j])
        outline_selected(ax[j], a, b)
        for zone in zones:
            outline_selected(ax[j], zone[0], zone[1], zoneName=zone[2])
    
    # Affichez les classes
    if liste_y is not None:
        for k in range(n):
            fig.text((k+0.5)/n, 0, '$r = $'+str(liste_y[k]), va='top', ha='center', fontsize=12)
    
    plt.subplots_adjust(left=0, right=1.1, top=1, bottom=0.2, wspace=0.05, hspace=0)
    plt.show()
    plt.close()

# Affichage de vingt images
def affichage_vingt(images):
    fig, ax = plt.subplots(2, 10, figsize=(10,2))
    for k in range(20):
        ax[k//10,k%10].imshow(images[k], cmap='gray')
        ax[k//10,k%10].set_xticks([])
        ax[k//10,k%10].set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()

# Affichage de trente images
def affichage_trente(images):
    fig, ax = plt.subplots(3, 10, figsize=(10,3))
    for k in range(30):
        ax[k//10,k%10].imshow(images[k], cmap='gray')
        ax[k//10,k%10].set_xticks([])
        ax[k//10,k%10].set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()

# Sauver le .csv

### VERSION 2/7 : on transforme les chiffres (2, 7) en classes (-1, 1) 
def sauver_et_telecharger_mnist_2(y_est_test, nom_du_fichier):
    y_est_test = np.array(y_est_test) # To array
    y_est_test_classes = np.where(y_est_test == 2, -1, np.where(y_est_test == 7, 1, y_est_test)) # From 2/7 to -1/1
    np.savetxt(nom_du_fichier, np.stack([ID_test_2, y_est_test_classes], axis=-1), fmt='%d', delimiter=',', header='ID,targets')
    basthon.download(nom_du_fichier)

def sauver_et_telecharger_mnist_4(y_est_test, nom_du_fichier):
    np.savetxt(nom_du_fichier, np.stack([ID_test_4, y_est_test], axis=-1), fmt='%d', delimiter=',', header='ID,targets')
    basthon.download(nom_du_fichier)

def sauver_et_telecharger_mnist_10(y_est_test, nom_du_fichier):
    np.savetxt(nom_du_fichier, np.stack([ID_test_10, y_est_test], axis=-1), fmt='%d', delimiter=',', header='ID,targets')
    basthon.download(nom_du_fichier)

# Visualiser les histogrammes
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

### VERSION 2 7

def afficher_histogramme(caracteristique, seuil = None, legende_cachee=False):
    # On vérife que 'seuil' est bien défini et n'est pas '...'
    if not check_is_defined(seuil):
        return

    c_train = [caracteristique(d) for d in d_train]        
    visualiser_histogrammes_mnist_2(c_train, pas = 10, mini_pas=2, x_min=0, grid=True, pas_y=50, seuil = seuil, legende_cachee=legende_cachee)

# Visualiser les histogrammes
def visualiser_histogrammes_mnist_2(c_train, x_min = None, x_max = None, y_max = None, pas_y = 100, legende_cachee = False, size='grand', legend_loc='upper right', legende_y = '\n Nombre d\'images\nde caractéristique $k$', thick_ticks=False, pas = 25, mini_pas = 5, grid=False, seuil=None):
    if size == 'grand':
        font_size = 14
    elif size == 'petit':
        font_size = 17
    else:
        raise ValueError("size doit valoir 'grand' ou 'petit'") 
    
    if x_min is None:
        x_min = int(min(c_train)/pas)*pas
    
    if x_max is None:
        x_max = int(max(c_train)/pas)*pas+pas
    
    nb_digits = len(chiffres)
    c_train_par_population = [np.array(c_train)[r_train==k] for k in classes]

    # Deux premières couleurs par défaut de Matplotlib
    colors = ['C0', 'C1']

    fig, ax = plt.subplots(figsize=(8,3))
    # Visualisation des histogrammes
    labels_hist = ['$h$', '$g$']

    # Création des bins :
    bins = np.arange(x_min, x_max+1, mini_pas)

    max_histo = []
    for k in range(nb_digits):
        #if legende_h_g:
        #label = labels_hist[k]+" ($y=$"+str(chiffres[k])+")"
        
        if legende_cachee:
            label = "$r=$ ?"
        else:
            label = "$r=$ "+str(chiffres[k])
        
        histo = ax.hist(c_train_par_population[k], bins=bins, alpha=0.7, density=False, 
            label=label, edgecolor='black')
        max_histo.append(max(histo[0]))
    
    if y_max is None:
        y_max = max(max_histo)+0.1*max(max_histo)

    ax.set_xlim(xmin=x_min)
    ax.set_xlim(xmax=x_max)
    ax.set_ylim(ymax = y_max)
    #ax.set_title("Histogrammes de la caractéristique")
    
    ax.legend(loc=legend_loc, fontsize=font_size+2)
    
    # Ticks principaux et secondaires sur x
    ax.xaxis.set_major_locator(MultipleLocator(pas))
    ax.xaxis.set_minor_locator(MultipleLocator(mini_pas))

    # Ticks principaux et secondaires sur y
    ecart = int((y_max//10)/pas_y)*pas_y+pas_y
    #ecart = 100
    ax.yaxis.set_major_locator(MultipleLocator(ecart))
    ax.yaxis.set_minor_locator(MultipleLocator(ecart//5))

    # Font size pour les ticks
    ax.tick_params(axis='both', labelsize=font_size)
    # Ticks épais si demandé:
    if thick_ticks:
        ax.tick_params(which='major', length=8, color='black', width=2.5)
        ax.tick_params(which='minor', length=5, color='black', width=1.5)

    # Enlever les axes de droites et du haut
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Centrer les axes en (0,0)
    ax.spines['left'].set_position(('data', x_min))
    ax.spines['bottom'].set_position(("data", 0))

    
    #Afficher les flèches au bout des axes
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(x_min, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)   
    
    # Nom des axex
    ax.set_xlabel('$k$', loc='right', fontsize=font_size)
    #if legende_h_g:
    #ax.set_ylabel('$h$ et $g$', loc='top', rotation='horizontal', fontsize=font_size)
    
    ax.set_ylabel(legende_y, rotation='horizontal', horizontalalignment='left', fontsize=font_size-4, labelpad=-180)
    ax.yaxis.set_label_coords(0.025, 0.95)

    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.5, which='major')
        
    if seuil is not None:
        global e_train
        erreur = 100*erreur_train_optim(c_train, r_train, seuil, classification)
        e_train = erreur

        # Show the error value as the title of the plot
        ax.set_title(f"Erreur associée à ce seuil = {erreur:.2f}%", fontsize=font_size)

        # Show the threshold as a vertical line
        ax.axvline(x=seuil, color='black', linestyle='--', linewidth=3)

    plt.tight_layout()
        

    plt.show()
    plt.close()
    #print(x_min, x_max, y_max)

# Visualiser les histogrammes
def visualiser_histogrammes_mnist_4(c_train_par_population):
    digits = [0,1,4,8]
    nb_digits = 4

    # Visualisation des histogrammes
    for k in range(nb_digits):
        plt.hist(np.array(c_train_par_population[k]), bins=60, alpha=0.7, label=digits[k], density = True)

    plt.gca().set_xlim(xmin=0)
    plt.gca().set_title("Histogrammes de la caractéristique")
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

# Visualiser les histogrammes 2D
def visualiser_histogrammes_2d_mnist_4(c_train):

    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()


# Visualiser les histogrammes 2D avec les domaines de Voronoi
def visualiser_histogrammes_2d_mnist_4_vor(c_train):
    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]
    theta = [np.mean(c_train_par_population[i], axis = 0) for i in range(4)]

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)

    fig, ax = plt.subplots(figsize=(10,10))

    # Voronoi
    vor = Voronoi(theta)
    fig = voronoi_plot_2d(vor, ax=ax, show_points=False)

    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

# Visualiser les histogrammes 2d avec les argmax des 4 distributions
def visualiser_histogrammes_2d_mnist_4_max(c_train):

    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]

    max_list = max_hist_2d_mnist4(c_train)

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(0,maxs_[0],100), np.linspace(0,maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(max_list[i][0], max_list[i][1], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()


# Visualiser les histogrammes 2D pour MNIST-10
def visualiser_histogrammes_2d_mnist_10(c_train):
    c_train_par_population = par_population_10(c_train)
    digits = np.arange(10).tolist()
    nb_digits = 10

    # Moyennes
    N_ = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N_[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N_[i] for i in range(nb_digits)]

    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", 'C'+str(i)], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor='C'+str(i))

    patches = [mpatches.Patch(color='C'+str(i), label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

# Visualiser les 10 hist 2D de MNIST-10 avec les domaines de voronoi
def visualiser_histogrammes_2d_mnist_10_vor(c_train):
    c_train_par_population = par_population_10(c_train)
    digits = np.arange(10).tolist()
    nb_digits = 10

    # Moyennes
    N_ = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N_[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N_[i] for i in range(nb_digits)]
    theta = [np.mean(c_train_par_population[i], axis = 0) for i in range(nb_digits)]

    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", 'C'+str(i)], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))

    # Voronoi
    vor = Voronoi(theta)
    fig = voronoi_plot_2d(vor, ax=ax, show_points=False)

    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor='C'+str(i))

    patches = [mpatches.Patch(color='C'+str(i), label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

# Trouve les coordonnées qui réalisent le max de chaque hist 2D de la caractéristique
def max_hist_2d_mnist4(c_train):
    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4
    max_list = []

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    for i in range(nb_digits):
        H, xedges, yedges = np.histogram2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
            bins=[np.linspace(0,maxs_[0],60), np.linspace(0,maxs_[1],60)])
        x, y = np.argwhere(H == H.max())[0]
        max_list.append([np.average(xedges[x:x + 2]), np.average(yedges[y:y + 2])])

    return max_list


# Fonction de score
def score(y_est, y_vrai):
    if len(y_est) != len(y_vrai):
        raise ValueError("Les sorties comparées ne sont pas de la même taille.")
    
    return np.mean(np.array(y_est) != np.array(y_vrai))

# Pour tracer la fonction erreur
from matplotlib.ticker import AutoMinorLocator

def tracer_erreur(func_classif, func_carac):

    pas_x = 4
    
    # Slice d_train and r_train using numpy's advanced slicing
    d_train_sliced = d_train[::pas_x]
    r_train_sliced = r_train[::pas_x]

    k_d_train_sliced = calcul_caracteristiques(d_train_sliced, func_carac)
    t_min = int(k_d_train_sliced.min())-1
    t_max = int(k_d_train_sliced.max())+1
    
    # Vectorize the erreur_train function to apply it over an array of t_values
    vec_erreur_train = np.vectorize(lambda t: 100 * erreur_train_optim(k_d_train_sliced, r_train_sliced, t, func_classif))
    
    # Calculer pas_t pour avoir environ 50 points:
    pas_t = int((t_max - t_min) / 50)+1
    # Create a range of t values using numpy's arange function
    t_values = np.arange(t_min, t_max, pas_t)

    # Apply the vectorized function to all t_values
    scores_array = vec_erreur_train(t_values)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.scatter(np.arange(t_min, t_max, pas_t), scores_array, marker='+', zorder=3)
    ax1.set_title("Erreur d'entrainement en fonction du paramètre seuil, MNIST 2 & 7")
    ax1.set_ylim((0, 68))
    ax1.set_xlim((t_min, t_max+2))
    ax1.set_xticks(np.arange(t_min, t_max, 2*pas_t))
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    # Enlever les axes de droites et du haut
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # Position des axes 
    if t_min <=0:
        # Dans ce cas, centrer les axes en (0,0)
        ax1.spines['left'].set_position(('data', 0))
        ax1.spines['bottom'].set_position(("data", 0))

        #Afficher les flèches au bout des axes
        ax1.plot(1, 0, ">k", transform=ax1.get_yaxis_transform(), clip_on=False)
        ax1.plot(0, 1, "^k", transform=ax1.get_xaxis_transform(), clip_on=False)  

    else:
        # Dans ce cas, centrer les axes en (t_min,0)
        ax1.spines['left'].set_position(('data', t_min))
        ax1.spines['bottom'].set_position(("data", 0))
        #Afficher les flèches au bout des axes
        ax1.plot(1, 0, ">k", transform=ax1.get_yaxis_transform(), clip_on=False)
        ax1.plot(t_min, 1, "^k", transform=ax1.get_xaxis_transform(), clip_on=False)

    
    # Nom des axex
    ax1.set_xlabel('Seuil $x$', loc='right')
    ax1.set_ylabel('$Erreur f(x)$', loc='top', rotation='horizontal')

    plt.tight_layout()
    plt.show()
    plt.close()
    
def tracer_erreur_c_train(t_min, t_max, c_train):
    pas_t = 2
    pas_x = 4
    scores_list = []
    for t in range(t_min, t_max, pas_t):
        e_train = 100*erreur_train(d_train[::pas_x], r_train[::pas_x], t, func_classif)
        scores_list.append(e_train)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.scatter(np.arange(t_min, t_max, pas_t), scores_list, marker='+', zorder=3)
    ax1.set_title("Erreur d'entrainement en fonction du paramètre seuil, MNIST 2 & 7")
    ax1.set_ylim(ymin=0, ymax=70)
    ax1.set_xticks(np.arange(t_min, t_max, 2*pas_t))
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()
    
# def tracer_erreur(t_min, t_max, func_classif):
#     pas_t = 2
#     pas_x = 4
#     scores_list = []
#     for t in tqdm(range(t_min, t_max, pas_t), desc='En cours de calcul... ', leave=True):
#         e_train = erreur_train(x_train[::pas_x], r_train[::pas_x], t, func_classif)
#         scores_list.append(e_train)

#     fig, ax1 = plt.subplots(figsize=(7, 4))
#     ax1.scatter(np.arange(t_min, t_max, pas_t), scores_list, marker='+', zorder=3)
#     ax1.set_title("Erreur d'entrainement en fonction du paramètre seuil, MNIST 2 & 7")
#     ax1.set_ylim(ymin=0, ymax=0.7)
#     ax1.set_xticks(np.arange(t_min, t_max, 2*pas_t))
#     ax1.xaxis.set_minor_locator(AutoMinorLocator())
#     ax1.yaxis.set_minor_locator(AutoMinorLocator())
#     plt.grid(which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.show()
#     plt.close()

# def tracer_erreur(s_min, s_max, pas, func_classif):
#     scores_list = []
#     for s in range(s_min, s_max, pas):
#         y_est_train = []
#         for x in x_train[::2]:
#             y_est_train.append(func_classif(x, s))
#         scores_list.append(score(y_est_train, r_train[::2]))

#     fig, ax1 = plt.subplots(figsize=(7, 4))
#     ax1.scatter(np.arange(s_min, s_max, pas), scores_list, marker='+', zorder=3)  # zorder=3 pour placer les croix au-dessus de la grille
#     ax1.set_title("Erreur d'entrainement en fonction du paramètre seuil, MNIST 1 & 6")
#     ax1.set_ylim(ymin=0, ymax=0.7)

#     # Pour afficher des valeurs plus précises sur l'axe x
#     ax1.set_xticks(np.arange(s_min, s_max, 5*pas))
    
#     # Pour ajouter des subticks sur l'axe x et y pour rendre la grille plus fine
#     ax1.xaxis.set_minor_locator(AutoMinorLocator())
#     ax1.yaxis.set_minor_locator(AutoMinorLocator())

#     plt.grid(which='both', linestyle='--', linewidth=0.5)  # Afficher les grilles principales et secondaires
#     plt.tight_layout()  # Ajuster l'affichage pour éviter tout chevauchement
#     plt.show()
#     plt.close()

from tqdm.notebook import tqdm

def grid_search(func, m_range, p_range, step_m, step_p, c_train):
    """
    Recherche les valeurs de m et p qui minimisent la fonction d'erreur.
    
    :param func: Fonction d'erreur à minimiser.
    :param m_range: Tuple indiquant la plage de recherche pour m.
    :param p_range: Tuple indiquant la plage de recherche pour p.
    :param step_m: Pas de recherche pour m.
    :param step_p: Pas de recherche pour p.
    
    :return: Tuple (m, p, min_error) où m et p minimisent la fonction d'erreur et min_error est l'erreur minimale.
    """
    
    m_values = np.arange(m_range[0], m_range[1], step_m)
    p_values = np.arange(p_range[0], p_range[1], step_p)
    
    min_error = 100
    best_m, best_p = None, None
    
    total_iterations = len(m_values) * len(p_values)
    pbar = tqdm(total=total_iterations, desc="Recherche du min ", dynamic_ncols=True)
    
    for m in m_values:
        for p in p_values:
            error = func(m, p, c_train, r_train)
            if error < min_error:
                min_error = error
                best_m = m
                best_p = p
                
            pbar.update(1)
    
    pbar.close()
                
    return round(best_m, 4), round(best_p, 4), round(min_error, 4)


""" m_range=(0, 2)
step_m=0.01

p_range=(1, 10)
step_p=0.5

best_m, best_p, min_err = grid_search(erreur_lineaire, m_range, p_range, step_m, step_p, c_train)
print(f"Meilleurs paramètres : m = {best_m}, p = {best_p} avec une erreur de {min_err}")
tracer_separatrice(best_m, best_p, c_train) """




# Moyenne
def moyenne(liste):
    arr = np.array(liste)
    return np.mean(arr)

def moyenne_zone(arr, a, b):
    if a is None or b is None:
        print_error("Les points A et B ne sont pas définis.")
        return 0
    
    numero_ligne_debut = min(a[0], b[0])
    numero_ligne_fin = max(a[0], b[0])
    numero_colonne_debut = min(a[1], b[1])
    numero_colonne_fin = max(a[1], b[1])
    return np.mean(arr[numero_ligne_debut:numero_ligne_fin+1, numero_colonne_debut:numero_colonne_fin+1])


def par_population_4(liste):
    chiffres = [0,1,4,8]
    # Créer une liste de liste qui divise par population, comme par exemple pour liste = c_train
    return [np.array(liste)[r_train_4==k] for k in chiffres]

def par_population_10(liste):
    # Créer une liste de liste qui divise par population, comme par exemple pour liste = c_train
    return [np.array(liste)[r_train_10==k] for k in range(10)]

def par_population_mnist_2(liste):
    # Créer une liste de liste qui divise par population, comme par exemple pour liste = c_train
    return [np.array(liste)[r_train_2==k] for k in chiffres]

def distance_carre(a,b):
    # a et b sont supposés être des points en deux dimensions contenus dans des listes de longueur deux
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def distance_carre_gen(A, B):
    return np.sum((A-B)**2)

def classification_2d_MNIST4(c, theta):

    #c_train_moyennes_par_population = [moyenne(liste_car) for liste_car in c_train_par_population]

    # On définit d'abord les différentes estimations possibles
    chiffres = [0,1,4,8]
    # On calcule le carré des distances entre la caractéristique c et les caractéristiques moyennes
    dist = [distance_carre(c, theta_i) for theta_i in theta]
    # On extrait l'indice minimisant cette distance
    index_min = dist.index(min(dist))
    # On renvoie le chiffre correspondant
    return chiffres[index_min]

# Algorithme de classification pour les 10 catégories de chiffres
def classification_dist_moy(c, theta_):
    # On définit d'abord les différentes estimations possibles
    chiffres = np.arange(10).tolist()
    # On calcule le carré des distances entre la caractéristique c et les caractéristiques moyennes
    dist = [distance_carre_gen(np.array(c).flatten(), np.array(theta).flatten()) for theta in theta_]
    # On extrait l'indice minimisant cette distance
    index_min = dist.index(min(dist))
    # On renvoie le chiffre correspondant
    return chiffres[index_min]

def visualiser_histogrammes_2d_mnist_2(c_train):

    c_train_par_population = par_population_mnist2(c_train)

    digits = [0,1]
    nb_digits = 2

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(mins_[0],maxs_[0],100), np.linspace(mins_[1],maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

def estim(d):
    if not has_variable('classification'):
        print_error("La fonction classification n'a pas été définie.")
        return False
    if not has_variable('caracteristique'):
        print_error("La fonction caracteristique n'a pas été définie.")
        return False
    if not has_variable('x'):
        print_error("La variable x n'a pas été définie.")
        return False
    
    caracteristique = get_variable('caracteristique')
    classification = get_variable('classification')
    x = get_variable('x')

    return classification(caracteristique(d), x)

def get_estimations(images, algorithme=None):
    if algorithme is None:
        algorithme = get_algorithme_func()
    if not algorithme:
        return None

    r_prediction = np.vectorize(algorithme, signature="(m,n)->()")(images)
    return r_prediction
    
def calculer_score(algorithme, method=None, parameters=None, cb=None, a=None, b=None):

    if sequence:
        set = d_test
    else:
        print("Mode DEV : Calcul du score sur d_train")
        set = d_train
    
    try:
        print("Calcul du score en cours...")

        def internal_cb(data):
            # Montre des exemples d'erreur avec d_train
            errors = []
            r_errors = []
            success = []
            r_success = []

            i = 0
            random_indices = np.random.permutation(len(d_train)) # Parcourt d_train dans un ordre random pour ne pas montrer toujours les mêmes exemples
            while (len(errors) < 5 or len(success) < 5) and i < len(random_indices):
                img_index = random_indices[i]
                if algorithme(d_train[img_index]) == r_train[img_index]:
                    if len(success) < 5:
                        success.append(d_train[img_index])
                        r_success.append(r_train[img_index])
                else:
                    if len(errors) < 5:
                        errors.append(d_train[img_index])
                        r_errors.append(r_train[img_index])
                
                i += 1
                    
            print("Voici quelques images que ton algorithme a mal classé :")
            affichage_dix(errors, liste_y=r_errors, n=len(errors), a=a, b=b)

            print("Et des images bien classées : ")
            affichage_dix(success, liste_y=r_success, n=len(success), a=a, b=b)

            if cb is not None:
                cb(data)

        r_prediction = get_estimations(set, algorithme=algorithme)
        if r_prediction is None:
            return 

        if not sequence:
            score = np.mean(r_prediction != r_train)
            set_score(score)
            internal_cb(score)
            return

        r_prediction_classes = np.where(r_prediction == 2, -1, np.where(r_prediction == 7, 1, r_prediction)) # From 2/7 to -1/1
        buffer = StringIO()
        np.savetxt(buffer, np.stack([ID_test_2, r_prediction_classes], axis=-1), fmt='%d', delimiter=',', header='ID,targets')
        buffer.seek(0)

        submit(buffer, method=method, parameters=parameters, cb=internal_cb)

    except Exception as e:
        print_error("Il y a eu une erreur lors du calcul du score. Vérifie ta réponse.")
        if debug:
            raise(e)

def calculer_score_etape_1():
    algorithme = get_algorithme_func()
    if algorithme is None:
        return
    
    def cb(score):
        validation_score_fixe()

    calculer_score(algorithme, method="fixed", cb=cb) 

def get_algorithme_func(error=None):
    if has_variable('algorithme'):
        algorithme = get_variable('algorithme')
    else:
        algorithme = None
        
    if algorithme is None or not callable(algorithme):
        if error:
            print_error(error)
        else:
            print_error("Vous avez remplacé autre chose que les ... . Revenez en arrière avec le raccourci clavier Ctrl+Z pour annuler vos modifications.")
        return None

    return algorithme
    
        
def check_pixel_coordinates(coords, errors):
    if not isinstance(coords, tuple) or len(coords) != 2:
        errors.append("Les coordonnées du pixel doivent être entre parenthèses séparées par une virgule. Exemple :")
        errors.append("(0, 0)")
        return False
    if coords[0] is Ellipsis or coords[1] is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if not isinstance(coords[0], int) or not isinstance(coords[1], int):
        errors.append("Les coordonnées du pixel doivent être des nombres entiers.")
        return False
    if coords[0] < 0 or coords[0] > 27 or coords[1] < 0 or coords[1] > 27:
        errors.append("Les coordonnées du pixel doivent être entre 0 et 27.")
        return False
    return True
    
### Validation cellules ###

def validate_algo(errors, answers):
    r_prediction = get_estimations(d_test)
    # check that all answers are 2 or 7
    if not np.all(np.logical_or(r_prediction == 2, r_prediction == 7)):
        non_matching = np.where(np.logical_and(r_prediction != 2, r_prediction != 7))
        first_non_matching = r_prediction[non_matching][0]
        if first_non_matching is Ellipsis:
            errors.append("Votre n'avez pas remplacé les ... dans la fonction algorithme")
        else:
            errors.append("Votre algorithme a répondu autre chose que 2 ou 7 : " + str(first_non_matching))
        return False
    
    return True 

def validation_func_score_fixed(errors, answers):     
    score_10 = answers['score_10']
    algorithme = get_algorithme_func(error="La fonction algorithme n'existe plus. Revenez en arrière et réexecutez la cellule avec 'def algorithme(d): ...'")
    if not score_10 or not algorithme:
        return False
    
    estimations = [algorithme(d) for d in d_train[0:10]]
    nb_errors = np.sum(estimations != r_train[0:10])

    if not isinstance(score_10, int):
        errors.append("La variable score_10 doit être un entier. Attention, écrivez uniquement le nombre sans le %.")
        return False

    if score_10 == nb_errors * 10:
        print(f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 premières images, soit {score_10}% d'erreur")
        return True
    
    if score_10 == nb_errors:
        errors.append("Ce n'est pas la bonne valeur. Tu as donné le nombre d'erreur et non le pourcentage d'erreur.")
    elif score_10 < 0 or score_10 > 100:
        errors.append("Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100.")
    else:
        errors.append("Ce n'est pas la bonne valeur. Note la réponse donnée par ton algorithme pour chaque image et compte le nombre de différences avec les bonnes réponses.")
    
    return False

def validate_pixel_noir(errors, answers):
    answers['pixel'] = int(d[17][15])
    if d[17][15] == 0:
        return True
    elif d[17][15] == 254:
        errors.append("Tu n'as pas changé la valeur du pixel, il vaut toujours 254")
    else:
        errors.append("Tu as bien changé la valeur mais ce n'est pas la bonne. Relis l'énoncé pour voir la valeur à donner pour un pixel noir.")
    return False
 
validation_algorithme_fixe = MathadataValidate(function_validation=validate_algo, success="Votre algorithme est valide. Il a bien répondu 2 ou 7 pour toutes les images !")
validation_execution_calcul_score = MathadataValidate(success="")
validation_question_score_fixe = MathadataValidateVariables({
    'score_10': None
}, function_validation=validation_func_score_fixed, success="")
validation_score_fixe = MathadataValidate(success="")
validation_execution_affichage = MathadataValidate(success="")
validation_question_pixel = MathadataValidateVariables({
    'pixel': {
        'value': int(d[15,17]),
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 255
                },
                'else': "Ta réponse n'est pas bonne. Les pixels peuvent uniquement avoir des valeurs entre 0 et 255."
            },
            {
                'value': int(d[17,15]),
                'else': "Attention ! Les coordonnées sont données en (ligne, colonne)."
            }
        ]
    }
})
validation_question_pixel_noir = MathadataValidate(success="Bravo, le pixel est devenu noir", function_validation=validate_pixel_noir)
validation_question_moyenne = MathadataValidateVariables({'k': np.mean(d[14:16,15:17])}, success="Bravo, la moyenne vaut en effet (142 + 154 + 0 + 0) / 4 = 74")