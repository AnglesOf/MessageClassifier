import joblib
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import configparser
import matplotlib.pyplot as plt
import seaborn as sns

from data_process import text_process, text2vector


def tarin_model(data, labels, textVectorizer,  model, modelName):
    """
    模型训练和保存
    :param data: 处理好后的训练数据列表
    :param labels: 标签列表
    :param textVectorizer: String，文本向量
    :param model: String，模型
    :param modelName: String，模型模型名称
    """
    # 按7:3分训练集和测试集
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(textVectorizer, labels,
                                                                                     data.index,
                                                                                     test_size=0.3, stratify=labels,
                                                                                     random_state=0)
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 对测试集进测试
    score = accuracy_score(y_pred, y_test)  # 计算准确率
    # print('accuracy %s' % score)
    # 保存模型
    # joblib.dump(model, saveModelFile)

    # 生成混淆矩阵
    plt.figure()
    conf_mat = confusion_matrix(y_test, y_pred)  # 混淆矩阵
    ticklabels = ["0", "1", "2", "3", "4", "5", "6"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=ticklabels, yticklabels=ticklabels)
    plt.ylabel('Actual results', fontsize=15)  # 设置X轴名称
    plt.xlabel('Prediction results', fontsize=15)  # 设置Y轴名称
    plt.title(modelName)  # 设置标题名称为模型名称
    plt.savefig(".\\results\\" + modelName + '_confusion_matrix.png')  # 保存混淆矩阵图片
    # plt.show()
    # 模型评估报告，包含了Precision，Recall和F1-Score
    report = classification_report(y_test, y_pred, target_names=["城乡建设", "环境保护", "交通运输", "教育文体", "劳动和社会保障", "商贸旅游", "卫生计生"])
    # print(report)
    return score, report


# 预测
def predictClassifier(text, modelFile, vectorizerFile, newdicFile, stopwordFile):
    """
    预测文档类别
    :param text:String，需要预测的文本
    :param modelFile: String，模型保存路径
    :param vectorizerFile: String，Vectorizer模型保存路径
    :param newdicFile: String，自定义字典文件路径
    :param stopwordFile: String，停用词文件路径
    :return: list[String]，预测结果
    """
    # 加载模型
    my_model = joblib.load(modelFile)
    tv = pickle.load(open(vectorizerFile, "rb"))
    # --------------------------------------------
    text_key = text_process(text, newdicFile, stopwordFile)
    test_features = tv.transform(text_key)
    return my_model.predict(test_features)


if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read('./fileConfig', encoding='utf8')
    #加载数据文件
    orgData = conf.get("orgFile", "orgdata")  # 原始数据路径
    newdicFile = conf.get("orgFile", "idictfile")  # 自定义字典路径
    stopwordsFile = conf.get("orgFile", "stopwordsfile")  # 停用词路径
    processed_date = conf.get("orgFile", "processed_date")  # 处理后的数据路径
    # 加载模型文件
    SVCFile = conf.get("model", "svcmodel")  # SVC存储路径
    BayesFile = conf.get("model", "bayesmodel")  # Bayes存储路径
    tfidf_vecmodel = conf.get("model", "tfidf_vecmodel")  # tfidf词向量存储路径
    bow_vecmodel = conf.get("model", "bow_vecmodel")  # BOW词向量存储路径

    # 数据处理,从excel读入数据
    # data_after_stop, labels = data_process(newdicFile, stopwordsFile, trainDataFile=orgData)  # 处理原始数据
    df = pd.read_excel(processed_date)  # 直接读取处理好的数据
    labels = df["labels"]  # 分类标签

    # 文本向量化
    textVectorizers = text2vector(df, tfidf_vecmodel, bow_vecmodel)  # 向量文本向量
    textVectorModelNames = ["TF-IDF", "BOW"]  # 文本向量模型名称

    # 加载模型
    SVCModel = LinearSVC(random_state=0, C=1.05, max_iter=4500)  # SVC模型
    NaiveBayesModel = MultinomialNB()  # 朴素贝叶斯模型
    # ModelFiles = [SVCFile, BayesFile]  # 模型存储路径
    modelNames = ["SVC", "NaiveBayes"]  # 模型名称
    models = [SVCModel, NaiveBayesModel]  # 模型列表

    # 训练-预测
    scores = []  # 准确率存储列表
    reports = []  # 模型评估报告存储列表
    for i in range(len(models)):
        for textVectoriser in textVectorizers:
            # 开始训练模型
            score, report = tarin_model(df, labels, textVectoriser, models[i], modelNames[i])
            print("模型：", modelNames[i], "\t向量化：", textVectorModelNames[i], "\taccuracy：", score)
            scores.append(score)
            reports.append(report)

