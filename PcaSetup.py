import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

df = pd.read_table('C:\Users\dcc\Desktop\DeepLearning\Tempus/tempus/takehome1.txt')     # 530 x 16563
Data = df.values
Data = scale(Data)
Xdata = Data[:,1:].copy()
GTdata = Data[:,0].copy()

components = [100,300,500,700]
i=1
for component in components:
    pca = PCA(n_components=component, whiten= True)
    pca.fit(Data)

    #The amount of variance that each PC explains
    var= pca.explained_variance_ratio_

    #Cumulative Variance explains
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    plt.figure(i)
    plt.plot(var1)
    plt.xlabel('Principal Components = %d' % component)
    plt.ylabel('Expained Varaince (%)')
    plt.title('Comparison')
    plt.show()
    i = i+1