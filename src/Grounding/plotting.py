from sklearn.manifold import MDS
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_space(X,y):
    fig = plt.figure(num="Knowledge Space",figsize=(8,8))
    for space_index,space_label in enumerate(X.keys()):
        X_current=X[space_label]
        y_current=y[space_label]
        ax = fig.add_subplot(2,2,space_index+1, projection="3d")
        ax.set_title(space_label)

        for point,label in zip(X_current,y_current):  
            ax.plot(point[0],point[1],point[2], "o")
            ax.text(point[0],point[1],point[2], label, size=10, zorder=1, color='k')

        axis_label={"x":"x", "y":"y", "z":"z"}    
        for i,label in axis_label.items():
            eval("ax.set_{:s}label('{:s}')".format(i, label))  


data_dir_grounding = os.path.dirname(__file__)
data_dir_knowledge = os.path.join(data_dir_grounding,"knowledge")

X={}
y={}
X_transformed={}

space_names = [ f.name for f in os.scandir(data_dir_knowledge) if f.is_dir() ] # ["color","shape","texture","general"]
for space_label in space_names:
    folder = os.path.join(data_dir_knowledge,space_label)
    knowledge_files = os.listdir( folder )
    if "X_"+space_label+".npy" in knowledge_files:
        X[space_label]=np.load(os.path.join(folder,"X_"+space_label+".npy"))
    if "y_"+space_label+".npy" in knowledge_files:
        y[space_label]=np.load(os.path.join(folder,"y_"+space_label+".npy"))           
    embedding = MDS(n_components=3)
    X_transformed[space_label] = embedding.fit_transform(X[space_label][:100])
    show_space(X_transformed,y)
plt.show()    


