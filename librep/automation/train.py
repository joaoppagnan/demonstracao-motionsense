# Classificação usando knn, dados brutos
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def plot_histogram(h, labels_dict=None, minv=None, maxv=None, min_max_v_xticks=10, output_path: str = None):
    """Plot a histogram using matplotlib.pyplot. h is a dictionary that maps each item to its frequency, 
       label_dic is a dictionary that maps each item value to a label, minv and maxv are used to draw a 
       red-dashed line on the histogram."""
    plt.grid(True, color = "lightgrey", linewidth = "0.8", linestyle = "--")
    if labels_dict is not None:
        plt.barh([f"{labels_dict[i]} - {i}" for i in h.keys()], h.values(), color='b')
    else:
        plt.barh([f"C: {i}" for i in h.keys()], h.values(), color='b')
        
    # Plot the min and max vertical bars and the labels
    if minv: plt.axvline(x=minv, ymin=0, ymax=1, color='r', linestyle="--")
    if maxv: plt.axvline(x=maxv, ymin=0, ymax=1, color='r', linestyle="--")
    if minv and maxv and min_max_v_xticks > 0:
        locs = [ int(minv+i*(maxv-minv)/min_max_v_xticks) for i in range(min_max_v_xticks+1) ]
        labels = [ "{}".format(int(i)) for i in locs ]
        plt.xticks(ticks=locs, labels=labels, rotation=90)
        
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', transparent=True)
        print(f"Figure saved to '{output_path}'")
        
    plt.show()
    plt.close()
    

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, labels, label_fmt_fnc = lambda x : str(x), desc: str = ""):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[ label_fmt_fnc(i) for i in labels])
    disp.plot(xticks_rotation='vertical')
    
def train_knn(X, y, n_neighbors: int=2):    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred

def train_svm(X, y, C: float=1.0, kernel: str = 'rbf'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = SVC(C=C, kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred    
    
def train_random_forest(X, y, n_estimators: int = 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred
#    print("KNN classifier - ", model)
#    print("F1-score:", "{0:.4f}".format(f1_score(y_test, y_pred, average='macro')))
#    print("Accuracy:", "{0:.4f}".format(accuracy_score(y_test, y_pred)))
#    print("Confusion Matrix")
#    plot_confusion_matrix(y_test, y_pred, model.classes_, label_fmt_fnc = lambda x : "{}-{}".format(kuhar.activity_names[x],x))