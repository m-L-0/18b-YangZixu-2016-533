```python
姓名:杨子旭
学号:2016011533
班级:16-5
```

## 任务:

* 根据已知9个类别的光谱样本,训练分类模型.
* 对2310个测试样本进行预测

## 思路:

* 读取数据集并对个样本标签**预处理成0-8,最后再改回来**
* **对数据集进行标准化处理(0均值,1标准差),**并划分测试集和验证集
* 训练xgboost模型,并调解优化参数
* 对模型进行训练,对验证集进行验证,并计算正确率
* 对测试集进行预测,**并转化成对应的类别标签**

## 代码部分:



### 导入各种库

```python
import scipy.io
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA,KernelPCA
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#print(data.values())
```

### 读取数据集

```python
L=[2,3,5,6,8,10,11,12,14]
num=[]
data=np.empty(shape=[0,200])
for i in L:
    tmp=scipy.io.loadmat('./train9/data'+str(i)+'_train.mat')['data'+str(i)+'_train']
    Len=tmp.shape[0]
    num.append(Len)
    data=np.append(data,tmp,axis=0)
print(data)
num=np.array(num)

x_train,x_test,y_train,y_test=train_test_split(data,values,test_size=0.2,random_state=5)
x_train=preprocessing.scale(x_train)
x_test=preprocessing.scale(x_test)
```

### 设置数据集对应的标签

```python
values=[]
for i in range(9):
    for j in range(num[i]):
        values.append(i)
values=np.array(values)
```

### 数据集划分成训练集和测试集,进行标准化预处理

```python
x_train,x_test,y_train,y_test=train_test_split(data,values,test_size=0.2,random_state=5)
x_train=preprocessing.scale(x_train)
x_test=preprocessing.scale(x_test)
```

### gridsearch 调xgboost参数

```python

'''
cv_params = {'max_depth':[x for x in range(1,20)],'min_child_weight':[x for x in range(1,10)], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
other_params = {'max_depth':10,'min_child_weight':2, 'gamma':0.3,'learning_rate': 0.1, 'seed': 123, 'subsample': 0.8,'n_estimators':9, 'colsample_bytree': 0.8,'silent':1,'alpha':0.05}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(x_train, y_train)
evalute_result = optimized_GBM.grid_scores_
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
'''
```

### 训练xgboost模型

```python
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 9,
    'gamma': 0.4,
    'max_depth': 11,
    'lambda': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight':3,
    'silent': 1,
    'eta': 0.1,
    'seed': 100,
    'n_estimators':9,
    'alpha':0.05,
    'learning_rate': 0.1
}

plst=params.items()
Dtrain=xgb.DMatrix(x_train,y_train)
num_round=42
model=xgb.train(plst,Dtrain,num_round)
```

### 计算正确率

```python
dtest = xgb.DMatrix(x_test)
ans = model.predict(dtest)
test_accuracy=accuracy_score(y_test,ans)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
```

### 读取测试集并进行预测

```python
Test=scipy.io.loadmat('./data_test_final.mat')['data_test_final']
Test=preprocessing.scale(Test)
DTest=xgb.DMatrix(Test)
ans=model.predict(DTest)
ANS=np.empty(shape=(len(ans),))
for i in range(len(ans)):
    ANS[i]=L[int(ans[i])]
#ANS 就是最后的结果集标签
print(ANS)
```







