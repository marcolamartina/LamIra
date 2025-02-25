from sklearn.manifold import MDS
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def show_spaces(X,y,legend=False):
    fig = plt.figure(num="Knowledge Space",figsize=(8,8))
    color={}
    for space_index,space_label in enumerate(X.keys()):
        X_current=X[space_label]
        y_current=y[space_label]
        ax = fig.add_subplot(2,2,space_index+1, projection="3d")
        ax.set_title(space_label)
        for point,label in zip(X_current,y_current):
            if label not in color.keys():
                color[label]=[random.random() for _ in range(3)]
                ax.text(point[0],point[1],point[2], label, size=10, zorder=1, color='k')
                if legend:
                    ax.plot(point[0],point[1],point[2], "o", color=color[label],label=label) 
            else:
                ax.plot(point[0],point[1],point[2], "o", color=color[label])
        if legend:
            ax.legend(loc='upper left')        
        axis_label={"x":"x", "y":"y", "z":"z"}    
        for i,label in axis_label.items():
            eval("ax.set_{:s}label('{:s}')".format(i, label))  


def show_space(X,y,space_label="general",legend=False):
    fig = plt.figure(num="Knowledge Space",figsize=(8,8))
    X_current=X[space_label]
    y_current=y[space_label]
    test_sets={ "general":["aglio","cipolla","spazzolino"],
                "color":["grigio","nero","rosso"],
                "shape":["semisfera","sfera","parallelepipedo"]}
    #test_set=["aglio","cipolla","spazzolino"]

    test_set=test_sets[space_label]
    #c={"rosso":(1,0,0),"nero":(0,0,0),"grigio":(0.5,0.5,0.5)}
    color={}
    ax = fig.add_subplot(1,1,1, projection="3d")
    ax.set_title(space_label)
    for point,label in zip(X_current,y_current):
        #if label not in test_set or (point[0]<-10 and point[1]>-2): #general
        #if label not in test_set or (point[0]>-0.05 and point[1]>-0.1 ) or point[2]>0.10: #color
        if label not in test_set or (point[0]<-10 and point[1]>-2): #general   
            continue
        if label not in color.keys():
            color[label]=[random.random() for _ in range(3)]
            #color[label]=c[label]
            #ax.text(point[0],point[1],point[2], label, size=10, zorder=1, color='k')
            if legend:
                ax.plot(point[0],point[1],point[2], "o", color=color[label],label=label) 
        else:
            ax.plot(point[0],point[1],point[2], "o", color=color[label])
    if legend:
        ax.legend(loc='upper left')        
    axis_label={"x":"x", "y":"y", "z":"z"}    
    for i,label in axis_label.items():
        eval("ax.set_{:s}label('{:s}')".format(i, label))

data_dir_grounding = os.path.dirname(__file__)
data_dir_base_knowledge = os.path.join(data_dir_grounding,"base_knowledge")
data_dir_plot = os.path.join(data_dir_grounding,"plot")



# Loading data
data={"X":{},"X_transformed":{},"y":{}}
filenames = [ f.name for f in os.scandir(data_dir_base_knowledge) if f.is_file() and f.name.endswith(".npy")] # ["color","shape","texture","general"]
for filename in filenames:
    space_label=filename.rsplit("_",1)[1][:-4]
    data_type=filename[0]
    data[data_type][space_label]=np.load(os.path.join(data_dir_base_knowledge,filename))  

# Loading data pre-computed
filenames_transformed = [ f.name for f in os.scandir(data_dir_plot) if f.is_file() and f.name.endswith(".npy")] 
for filename in filenames_transformed:
    space_label=filename.rsplit("_",1)[1][:-4]
    data["X_transformed"][space_label]=np.load(os.path.join(data_dir_plot,filename))
'''
# MDS
embedding = MDS(n_components=3)
for space_label,X in data["X"].items():
    data["X_transformed"][space_label] = embedding.fit_transform(X)
    np.save(data_dir_plot+"/X_trasformed_"+space_label+".npy",data["X_transformed"][space_label]) 
'''    

# Plotting
show_space(data["X_transformed"],data["y"],legend=True,space_label="general")
plt.show()

 


