GeneticDrawing

Algorithmes génétiques pour la création de peintures à partir de portraits

Description : Une "peinture" est définie ici comme une superposition de "coups de pinceau", chacun caractérisé par une position donnée, une direction, une couleur, une opacité et une taille. Le problème à résoudre peut être formulé ainsi : Étant donné une image de référence, un canevas de dimensions fixes et un nombre limité de coups de pinceau, comment les agencer pour produire une peinture ressemblant à l’image de référence ? Ce sujet s’inscrit dans la continuité des travaux d’Anastasia Opara (voir un exemple ici : https ://github.com/anopara/genetic-drawing), qui a développé une approche basée sur des algorithmes génétiques pour générer des peintures en noir et blanc. Les objectifs de cette étude sont les suivants :

Étendre cette approche génétique à la création de peintures en couleurs.
Intégrer des outils standard de traitement d’image (tels que le filtre de Sobel ou la détection de points d’intérêt) pour optimiser la génération et la disposition des coups de pinceau.
Exploiter des algorithmes génétiques plus avancés, disponibles via la librairie DEAP, afin d’accélérer les rendus.
Permettre à l’utilisateur de comparer les résultats obtenus et concevoir un système adaptatif capable de sélectionner les coups de pinceau les mieux adaptés à ses préférences. L’objectif final est de développer une application en Python permettant de transformer un portrait au format "selfie" en une peinture. Cette application devra également inclure une fonctionnalité d’amélioration incrémentale, où l’utilisateur pourra comparer deux rendus à chaque étape et choisir celui qui correspond le mieux à ses attentes.
