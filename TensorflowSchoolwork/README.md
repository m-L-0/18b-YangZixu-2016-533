# :) 抄袭，一律按0分计算。

```
姓名:杨子旭
班级:16-5
学号:2016011533
```

## TensorFlow School Work

#### 任务列表

1. 将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集（数据集部分可使用Dataset API，也可以不使用）。
2. 设计模型：
   * 使用TensorFlow设计K近邻模型（可不使用KD树优化算法）
   * 模型关键部分需添加注释
3. 训练模型：
   * 使用TensorFlow完成训练相关的代码
   * 训练关键部分需添加注释
4. 验证模型：
   * 使用验证集检测模型性能
   * 使用验证集调整超参数
5. 提交模型
   * 可使用Eager模式设计模型
   * 提交文件必须为执行后的 Jupyter Notebook 文件
   * 文件中必须包含模型代码、训练代码、模型性能评估代码、最终在验证集上的结果、关键部分注释等内容。

代码部分:

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle 

```

### 加载数据集并打乱随机划分

```python
iris=load_iris()
data=iris.data
print(data.shape)
value=iris.target
data,value = shuffle(data,value)
#随机打乱划分按照训练集:验证集=8:2 划分
train_idx, val_idx = next(iter( StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10).split(data,value)))
#按照返回的测试集和验证集的下标,从原来数据进行提取
x_train=data[train_idx]
y_train=value[train_idx]
x_test=data[val_idx]
y_test=value[val_idx]
```

### 选择占位符

```python
#X1 代表训练集
X1=tf.placeholder("float",[120,4])
#X2 代表验证集的一个数据
X2=tf.placeholder("float",[4])
#K代表选择近邻的个数
K=tf.placeholder("int32")
```

### 计算欧氏距离

```python
#计算到验证集点到每一个训练集点的距离
dist=tf.reduce_sum(tf.sqrt(tf.pow(X1-X2,2)),reduction_indices=1)
#选取所有所有距离的从大到小的索引值
pred=tf.nn.top_k(dist,120)
```

### 分类精确度

```python
accuracy=0.
```

### 运行会话,训练+验证模型

```python
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #print(x_train)
    for m in range(1,11):
        accuracy = 0.
        for i in range(30): 
            nn_index=sess.run(pred,feed_dict={X1:x_train,X2:x_test[i,:],K:m})
            #print(nn_index.values)
            #print(nn_index.indices)
            Count={0:0,1:0,2:0}
            Tmp=nn_index.indices[-m:,] #选取距离最小的m个索引
            for j in Tmp:#选择其中数量最多的类别作为此样本的类别
                Count[y_train[j]]+=1
            key_name=max(Count,key=Count.get)
            if key_name == y_test[i]:#计算准确率
                accuracy += 1.0
        print("K值为"+str(m)+"时精确率:%.2f%%" % (accuracy/30*100))
```

### 最终结果

```python
K值为1时精确率:100.00%
K值为2时精确率:90.00%
K值为3时精确率:96.67%
K值为4时精确率:93.33%
K值为5时精确率:96.67%
K值为6时精确率:93.33%
K值为7时精确率:96.67%
K值为8时精确率:96.67%
K值为9时精确率:96.67%
K值为10时精确率:96.67%
    
所以K=1时,即为最近邻时效果最好
```



