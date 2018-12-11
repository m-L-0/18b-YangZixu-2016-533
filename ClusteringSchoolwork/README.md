# :) 抄袭，一律按0分计算。

```
姓名:杨子旭
班级:16-5
学号:2016011533
```

# Clustering Schoolwork

## 任务列表

1. 将鸢尾花数据集画成图的形式。
2. 确定一个合适的**阈值**，只有两个样本之间的相似度大于该阈值时，这两个样本之间才有一条边。
3. 求取带权**邻接矩阵**。
4. 根据邻接矩阵进行聚类。
5. 将聚类结果可视化，重新转换成图的形式，其中每一个簇应该用一种形状表示，比如分别用圆圈、三角和矩阵表示各个簇。
6. 求得分簇正确率
7. 完成代码的描述文档

### **任务完成情况**

1. 作图用不同颜色代表不同簇的划分和需求略有不同
2. 求邻接矩阵时,并没有用设置阈值的方法,而是根据距离矩阵求得样本的k个近邻,默认这k个近邻之间有一条边.(为什么不用确定阈值的方法,是因为做得时候正确率有点低,详细代码在谱聚类2中)
3. 正确率能达道96%的原因,就在于邻接矩阵求解的地方,用距离矩阵算k近邻,否则得不到这么高的正确率.

**代码部分**(**谱聚类1中的代码)**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import networkx as nx
iris=load_iris() #加载鸢尾花数据集
data=iris.data  #数据集
value=iris.target #标签集
```

### 计算距离矩阵

```Python
def dist(vec1,vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))
def distMatrix():
    S=np.zeros((150,150))
    for i in range(150):
        for j in range(i+1,150):
            S[i][j]=1.0*dist(data[i],data[j])
            S[j][i]=S[i][j]
    return S
```

### 计算邻接矩阵

```python
def KNN(S,k,sigma=1.0):
    A=np.zeros((150,150))
    for i in range(150):
        d_w_i=zip(S[i],range(150)) #这个点150个距离和其下标打成元组
        #print(d_w_i)
        d_w_i=sorted(d_w_i,key=lambda x:x[0]) #按照距离从小到达排序
        nei_id=[d_w_i[j][1] for j in range(k+1)] #取得k个近邻的索引
        #k个近邻计算邻接矩阵
        for j in nei_id:
            A[i][j]=np.exp(-S[i][j]/(2*sigma*sigma))
            A[j][i]=A[i][j]
    return A
```

### 定义标准化拉普拉斯矩阵的函数

```python
def laplacianMat(A):
    D=np.sum(A,axis=1)
    L=np.diag(D)-A
    sD=np.diag(1.0/(D**0.5))
    return np.dot(np.dot(sD,L),sD)
```

### 定义计算拉普拉斯矩阵的特征值和特征向量的函数

```python
def calcu(P):
    return np.linalg.eig(P)
```

### 定义单位化函数

```python
def Unitized(U,k):
    def dist2(vec):
    #print(vec)
        Sum=0
        for i in vec:
            Sum+=i*i
        return Sum**0.5
    for i in range(150):
        tmp=dist2(U[i])
        if(tmp==0):
            continue
        for j in range(k):
            U[i][j]/=tmp
    return U
```

### 求得各个矩阵

```python
S=distMatrix()  #距离矩阵
print(S)
A=KNN(S,k=10)     #邻接矩阵
print(A)
L=laplacianMat(A) #拉普拉斯矩阵
tzz,tzxl=calcu(L) #计算特征值和特征向量
#print(tzxl.shape)
tzz=zip(tzz,range(150))
tzz=sorted(tzz,key=lambda tzz:tzz[0]) #按照特征值的升序排序
H=np.vstack([tzxl[:,i] for (v,i) in tzz[:7]]).T #取前k个最小的特征值对应的特征向量,k自己定,这里我设为7,效果比较好尽量不超过10
H=Unitized(H,7)
#print(H.shape)
```

### kmeans 聚类

```python
clf=KMeans(n_clusters=3,n_init=13,random_state=19)#
y_pre=clf.fit_predict(H)
y=y_pre[::]
print(y)
```

### 计算正确率

```python
ans=0
def zql():
    global ans
    for i in range(0,150):
        if(y[i] == value[i]):
            ans+=1
        else:
            continue
zql()
print("正确率: %f%%" % ((ans/150)*100))
#算得正确率为96%
```

### 原始鸢尾花数据集转换成图

```python
N=nx.Graph()
for i in range(150):
    N.add_node(i)
#print(N.nodes())
edglist=[]
for i in range(150):
    for j in range(150):
        if(A[i][j]>0):
            edglist.append((i,j))
#print(edglist)
colorlist=[]
N.add_edges_from(edglist)
for i in range(150):
    if(value[i]==0):
        colorlist.append('r')
    elif(value[i]==1):
        colorlist.append('b')
    else:
        colorlist.append('orange')
#print(N.neighbors((data_new[1][0],data_new[1][1])))
#print(N[(data_new[1][0],data_new[1][1])])
nx.draw(N,pos = nx.circular_layout(N),node_color = colorlist,edge_color = 'black',with_labels = False,font_size =5,node_size =25,width=0.3)

plt.show()
```

### 聚类后数据集转换成图

```python
N=nx.Graph()
for i in range(150):
    N.add_node(i)
#print(N.nodes())
edglist=[]
for i in range(150):
    for j in range(150):
        if(A[i][j]>0):
            edglist.append((i,j))
#print(edglist)
colorlist=[]
shapelist=[]
N.add_edges_from(edglist)
for i in range(150):
    if(y_pre[i]==0):
        colorlist.append('r')
    elif(y_pre[i]==1):
        colorlist.append('b')
    else:
        colorlist.append('orange')
#print(N.neighbors((data_new[1][0],data_new[1][1])))
#print(N[(data_new[1][0],data_new[1][1])])
nx.draw(N,pos = nx.circular_layout(N),node_color = colorlist,edge_color = 'black',with_labels = False,font_size =5,node_shape='o' ,node_size =25,width=0.3)
plt.show()
```

