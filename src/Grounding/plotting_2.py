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
    points={"aglio":[95.09267013399148, -1.3681241184767294, -2.3244005641748915, 95.09267013399148, -1.3681241184767294, -2.3244005641748915, 65.35998233215548, -1.3074204946996484, -7.289752650176686, 0.6703877790834313, 0.8343125734430082, 0.20067640335750928, 2.196675143502039, 0.7923380097309888, 0.35118497999384424, -2.6132840353130184, -0.0056123248283569535, 0.48496594154730177, 1.7294831532499466, 0.4523264767297567, 4.999623331103596, 0.9580652448566519, 0.3308228870601847, 0.7762232374771356, 0.7474724045447728, 1.2955158630679413, 3.189158006992921, 3.733989726658287, 0.002197246173956103, 2.1607867955113145, -0.6401459183345197, 0.9849152768966902],
            "cipolla":[90.24803755961844, -2.449920508744038, -3.588235294117645, 67.0229473431498, -1.3854033290653014, -5.016645326504478, 37.0754665798611, -1.229166666666667, 0.7934027777777728, 0.7515151515151515, 0.8750733137829912, 0.2352995567569597, 2.3214845970320277, 0.5347173674867586, 0.4629333333333333, -2.524296839571263, -0.004300492846917566, 0.4840953151032503, 1.7489559689309937, 0.30976390197848497, 2.186561648513044, 0.9742479860000401, 0.18860486214003788, 0.6364335056589286, 0.7043485064847391, 0.7447013967889825, 4.716855037873139, 5.957710921968878, 0.0014047587020134076, 2.862153410229845, -0.5747559619028677, 0.9958638474150826],
            "spazzolino":[34.42402906378602, 1.1049382716049387, -2.3786008230452715, 37.373218201754405, 3.9144736842105132, -30.100877192982473, 61.19540662650601, -1.4216867469879617, -14.619277108433739, 0.20053199349785725, 0.6519402985074627, 0.09228238990384177, 2.3198869757426146, 0.6072602597106166, 0.11013716419121825, -2.235966771640067, -0.005929838608272888, 0.468110113580501, 1.4025620175064513, 0.6758730130851908, 1.5903907221186824, 0.9303462330241659, 0.06748844481396436, 0.8482340260185064, 0.21839510980024798, 0.2605431832669895, 2.063639378255023, 2.526661564861765, 0.002854440133672476, 1.5370429008034177, -0.5655794729814563, 0.9286022222236044],
            "tastiera":[2.498721810745085, -1.0217112535927266, 0.03749723634754665, 2.498721810745085, -1.0217112535927266, 0.03749723634754665, 9.787407063197033, -0.3742255266418807, -8.097273853779413, 0.6046101694915255, 0.712180790960452, 0.12110771587299538, 2.3591466817751985, 0.5674044270808517, 1.4679835390946503, -2.0978866225254555, -0.020111948888960663, 0.4888574833933601, 1.833381041187454, 0.20612098769630594, 0.41290978196871486, 0.8516277329390385, 0.009801521772753103, 0.6605775960785563, 0.1019425559714459, 0.036298271725035545, 4.260834644074278, 5.783709622303453, 0.0023336187399720246, 2.5681126310688134, -0.4268831704617797, 0.977392046170917]}
    
    embedding = MDS(n_components=3)
    a=list(points.values())
    b=list(points.keys())

    X=np.append(data["X"]["general"],np.array([a[0],a[1],a[2],a[3]]), axis=0)
    
    X_current=data["X_transformed"][space_label]
    y_current=y["general"]
    y_current=np.append(y_current,np.array([b[0],b[1],b[2],b[3]]), axis=0)

    #X_current= embedding.fit_transform(X)
    #np.save(data_dir_plot+"/X_trasformed_points.npy",X_current)
    fig = plt.figure(num="Knowledge Space",figsize=(8,8))
    
    test_sets={ "general":["aglio","cipolla","spazzolino","tastiera"],
                "color":["grigio","nero","rosso"],
                "shape":["semisfera","sfera","parallelepipedo"],
                "points":["aglio","cipolla","spazzolino","tastiera"]}
    #test_set=["aglio","cipolla","spazzolino"]
    test_set=test_sets[space_label]
    c={"aglio":(1,0.9,0),"cipolla":(0,1,0),"spazzolino":(1,0,0),"tastiera":(0,0,1)}
    color={}
    ax = fig.add_subplot(1,1,1, projection="3d")
    ax.set_title(space_label)
    g=len(X_current)
    for j,(point,label) in enumerate(zip(X_current,y_current)):
        print("{:.2f}%".format(j/g),end="\r")
        #if label not in test_set or (point[0]<-10 and point[1]>-2): #general
        #if label not in test_set or (point[0]>-0.05 and point[1]>-0.1 ) or point[2]>0.10: #color
        if label not in test_set: #general   
            continue
        if label not in color.keys():
            color[label]=[random.random() for _ in range(3)]
            color[label]=c[label]
            #ax.text(point[0],point[1],point[2], label, size=10, zorder=1, color='k')
            if legend:
                ax.plot(point[0],point[1],point[2], "o", color=color[label],label=label) 
        else:
            if j>len(X_current)-5:
                ax.plot(point[0],point[1],point[2], "o", color=(0,0,0))
                print(label,point[0],point[1],point[2])
            else:    
                ax.plot(point[0],point[1],point[2], "o", color=color[label])
                
    #edgecolor='black'
    #for lab,p in points.items():

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
'''
l=["aglio","cipolla","spazzolino","tastiera"]
f=np.where((data["y"]["general"]==l[0])|(data["y"]["general"]==l[1])|(data["y"]["general"]==l[2])|(data["y"]["general"]==l[3]))
data["X"]["general"]=data["X"]["general"][f]
data["y"]["general"]=data["y"]["general"][f]
'''
show_space(data["X_transformed"],data["y"],legend=True,space_label="points")
plt.show()

 


