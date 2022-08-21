# 作者：李幸阜

**本项目为2022年8月研究生入学前暑假培训AI项目**

## 1、实验环境

### 1.1、硬件环境：

| 环境 |    版本/型号     |
| :--: | :--------------: |
| cpu  | Intel i5 8th Gen |
| gpu  |     GTX1050      |
| cuda |     10.1.243     |

### 1.2、软件环境：

|     环境     | 版本/型号 |
| :----------: | :-------: |
|    python    |    3.8    |
|   pytorch    |   1.7.1   |
|   sklearn    |  0.24.1   |
|    pkuseg    |  0.0.25   |
|    pandas    |  0.23.4   |
| configparser |   5.0.2   |

## 2、文件说明

​	该项目所有数据预处理的文件路径均在config文件中进行配置

​	*model文件夹* 下保存词向量模型和分类模型，由于GRU模型太大因此未上传

​	*results文件夹* 下保存输出文件，包含混淆矩阵和GRU训练ACC随EPOCH的变化

​	*data文件夹* 下包含了原始数据和信息、分词、去除停用词后的数据，以及自定义词典和停用词库。

​	*data_process.py* 包含了数据预处理的一些函数、算法

​	*ML_TextClassifier.py* 包含了SVM分类模型和朴素贝叶斯分类模型

​	*GRUClassifier.py* 定义了GRU分类模型，*GRUClassifier.ipynb* 是它的 jupyter notebook 版本。

​	*基于自然语言处理的留言分类算法.pdf* 是该项目下的作业论文

## 3、系统架构图

​	首先对数据进行清洗、分词、去除停用词后，使用TF-IDF、词袋模型、word2vec工具对文本进行向量化，使用TF-IDF矩阵和BOW矩阵训练SVM模型和朴素贝叶斯模型，使用word2vec工具生成的embedding训练GRU模型，之后对这三种模型进行评价和分类效果对比。

![image-20220821205559352](https://github.com/AnglesOf/MessageClassifier/blob/master/results/image-20220821205559352.png)

## 4、GRU分类模型架构图

​	GRU神经网络分类模型共分为Embedding层，GRU层，全连接层。首先将每个文本信息的one-hot编码作为Embedding层的输入，经过CBOW模型训练后生成每个词的词向量，one-hot编码的维度为83658，因此Embedding输入维度为83658，输出维度设为512。这将大大降低词向量的维度，方便进行下游任务。之后将这些词向量作为GRU层的输入，GRU的输入维度为512，这与词向量维度保持一致，输出维度为128，利用这些时间序列训练GRU单元得到隐藏层输出$h_n$，然后再将$h_n$作为全连接层的输入，全连接层的输出维度和类型个数相同，因此设置为7维，最后利用交叉熵损失函数计算损失，计算出损失值后进行反向传播更新模型参数。模型具体表述如图 4所示。

<img src="https://github.com/AnglesOf/MessageClassifier/blob/master/results/image-20220821210025849.png" alt="image-20220821210025849" style="zoom:150%;" />

## 5、分类结果报告

|                        | precision | recall | f1-score | ACC  |
| :--------------------: | :-------: | :----: | :------: | :--: |
|     SVM （TF-IDF）     |   0.91    |  0.89  |   0.90   | 0.91 |
|      SVM （BOW）       |   0.87    |  0.87  |   0.87   | 0.87 |
| Naive Bayes （TF-IDF） |   0.79    |  0.68  |   0.63   | 0.68 |
|  Naive Bayes （BOW）   |   0.86    |  0.86  |   0.86   | 0.86 |
|          GRU           |   0.81    |  0.81  |   0.81   | 0.81 |
