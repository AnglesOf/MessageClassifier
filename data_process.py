import pickle
import re
import configparser
import jieba
import jieba.analyse
import pandas as pd
import pkuseg


# 清洗数据
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def clean_text(text):
    '''
    清洗数据，对internetURL、域名、超过5个的数字、日期、网址、年、月、日、敬语词、特殊字符都删除
    :param text:输入句子
    :return: 过滤后的句子，例如"A市15271020426安全隐患https请求查处"，过滤后为"A市安全隐患请求查处"
    '''
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    # text = text.replace('，'+'。'+'/', " ")  # 分隔符，我们是不需要的
    # text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。
    text = re.sub(r'((https|http|ftp|rtsp|mms):/{2})[www.]*[a-z0-9]{1,}.(com|cn)(/[a-z0-9]{1,}(.[a-z]{1,})*){0,}', '',
                  text)  # internetURL
    text = re.sub(r'[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(/.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+/.?', '', text)  # 域名
    # text = re.sub(r'\d{15}|\d{18}', '', text)  # 身份证号
    # text = re.sub(r'[1]([3-9])[0-9]{9}', '', text)  # 手机号
    text = re.sub('[0-9]{5,}', '', text).strip()  # 超过5个的数字
    text = re.sub(r"\d+/\d+/\d+", "", text)  # 日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)  # 时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text)  # 邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)  # 网址，没意义
    text = re.sub("['年', '月', '日', '尊敬', '书记', '领导']", "", text)  # 年、月、日、敬语没意义
    text = re.compile(
        u"[\u3002\uFF1F\uFF01\uFF0C\u3001\uFF1B\uFF1A\u300C\u300D\u300E\u300F\u2018\u2019\u201C\u201D\uFF08\uFF09\u3014\u3015\u3010\u3011\u2014\u2026\u2013\uFF0E\u300A\u300B\u3008\u3009\!\@\#\$\%\^\&\*\(\)\-\=\[\]\{\}\\\|\;\'\:\"\,\.\/\<\>\?\/\*\+]+").sub(
        '', text)
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text


# 加载停用词list
def stopwordslist(stopwordsPath):
    stopwords = [line.strip() for line in open(stopwordsPath, 'r', encoding='utf-8').readlines()]
    stopwords.append('省')
    stopwords.append('县')
    stopwords.append('你好')
    stopwords.append('您好')
    stopwords.append('请问')
    stopwords.append('尊敬')
    stopwords.append('领导')
    stopwords.append('天')
    stopwords.append('太')
    stopwords.append('时')
    stopwords.append('未')
    stopwords.append('次')
    stopwords.append('名')
    stopwords.append('元')
    stopwords.append('两')
    stopwords.append('https')
    stopwords.append('http')
    for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']:
        stopwords.append(i)
    return stopwords


# 对句子进行分词
def seg_sentence_list_to_string(sentence, seg, stopwords):
    '''
    中文分词
    :param sentence: 句子文本
    :param seg: 分词器，为None时使用jieba分词，不为None使用pkuseg分词
    :param stopwords: 停用词
    :return: 分词后的句子，例如:"A市湖建筑集团占道施工有安全隐患"，分此后:"A市 西湖 建筑 集团 占 道 施工 安全隐患"
    '''
    if sentence.strip() == '':
        return ''
    if seg == None:
        sentence_seged = jieba.cut(sentence.strip())
    else:
        sentence_seged = seg.cut(sentence.strip())
    # jieba.load_userdict(newdicFile)
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr.strip()


# 对句子进行分词
def seg_sentence_to_string(sentence, newdicFile, stopwordsPath):
    '''
    中文分词
    :param sentence: 句子文本
    :param newdicFile: 自定义字典路径
    :param stopwordsPath: 停用词路径
    :return: 分词后的句子，例如:"A市湖建筑集团占道施工有安全隐患"，分此后:"A市 西湖 建筑 集团 占 道 施工 安全隐患"
    '''
    if sentence.strip() == '':
        return ''
    seg = pkuseg.pkuseg(user_dict=newdicFile)
    sentence_seged = seg.cut(sentence)
    # jieba.load_userdict(newdicFile)
    # sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist(stopwordsPath)  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr.strip()


# 将xlsx表的某一列转化为txt文本
def titleToTxt(columns, dataFile="附件二.xlsx", txtFile="附件4.txt"):
    df = pd.read_excel(dataFile)
    df = df[columns]
    with open(txtFile, 'w+', encoding='utf-8') as f:
        df.apply(lambda x: f.write(x + '\n'))
    f.close()


def text_process(text, newdicFile, stopwordFile)->list:
    """
    单条数据预处理
    :param text:String，需要预测的文本
    :param newdicFile: String，自定义字典文件路径
    :param stopwordFile: String，停用词文件路径
    :return: list[String]，预测文本的关键词列表，例如:"江夏大道武汉纺织大学"，return为["江夏 武汉 纺织 大学"]
    """
    text = text.strip()
    text = clean_text(text)
    text = seg_sentence_to_string(text, newdicFile, stopwordFile)
    # keywords = jieba.analyse.extract_tags(text, topK=30)
    # key = [' '.join(keywords)]
    key = [' '.join(text)]
    return key


def data_process(newdicFile, stopwordFile, trainDataFile):
    """
    训练数据处理
    :param trainDataFile: String，训练数据路径，当为None时从数据库读入，否则从本地读入
    :param newdicFile: String，自定义字典文件路径
    :param stopwordFile: String，停用词文件路径
    :return:
    """
    if trainDataFile is None:
        raise Exception("训练数据路径未知！")
    elif newdicFile is None:
        raise Exception("字典据路径未知！")
    elif stopwordFile is None:
        raise Exception("停用词路径未知！")
    else:
        df = pd.read_excel(trainDataFile)
        seg = pkuseg.pkuseg(user_dict=newdicFile)
        # jieba.load_userdict(newdicFile)
        stopwords = stopwordslist(stopwordFile)  # 这里加载停用词的路径
    d = df['cat'].value_counts()
    labels = ["Urban and \nrural construction", "Environmental \nprotection",
              "Health and \nfamily planning", "Transportation", "Educational \nstyle",
              "Labor and social \nsecurity", "Business \ntourism"]
    plt.figure(figsize=(12, 8))  # 设置画布大小
    plt.title("Quantity distribution")
    plt.ylabel('Quantity', fontsize=18)
    plt.xlabel('category', fontsize=18)
    plt.tick_params(axis='x', labelsize=11)  # 设置x轴标签大小
    plt.bar(labels, d)
    plt.show()
    df.drop_duplicates(subset=['留言详情'], keep='first', inplace=True)
    # 处理留言详情
    df['留言详情'] = df['留言详情'].apply(lambda s: s.strip())
    df['留言详情clean'] = df['留言详情'].apply(lambda s: clean_text(s))
    df['留言详情clean'] = df['留言详情clean'].apply(lambda s: seg_sentence_list_to_string(s, seg, stopwords))
    # 处理标题
    df['review'] = df['review'].apply(lambda s: s.strip())
    df['reviewclean'] = df['review'].apply(lambda s: clean_text(s))
    df['reviewclean'] = df['reviewclean'].apply(lambda s: seg_sentence_list_to_string(s, seg, stopwords))

    df['textData'] = df['留言详情clean'] + df['reviewclean']  # 留言和标题合并
    df['textData'] = df['textData'].apply(lambda x: " ".join(list(set(x.split(" ")))))  # 去除重复分词
    # 处理留言时间
    # df['留言时间'] = df['留言时间'].apply(lambda x: re.sub(r'/', '-', x))
    # key = []
    # for i in df['留言详情clean']:
    #     keywords = jieba.analyse.extract_tags(i, topK=30)
    #     key.append(keywords)
    #     # key.append(' '.join(keywords))
    # df['key'] = key
    labels = df.loc[:, 'cat'].factorize()[0]  # 将标签进行编号
    return df, labels


def text2vector(data, tfidfVectorizerFile, countVectorizerFile):
    adata_set = data['textData']

    # TF-IDF文本向量化
    tfidfVectorizer = TfidfVectorizer(norm='l2', ngram_range=(1, 1))
    tfidfFeature = tfidfVectorizer.fit_transform(adata_set)

    # 词袋文本向量化
    countVectorizer = CountVectorizer(min_df=5, max_df=0.6)
    BOWFeature = countVectorizer.fit_transform(adata_set)

    # 存储tfidfVectorizer模型
    pickle.dump(tfidfVectorizer, open(tfidfVectorizerFile, "wb"))
    # 存储countVectorizer模型
    pickle.dump(countVectorizer, open(countVectorizerFile, "wb"))

    return [tfidfFeature, BOWFeature]


if __name__ == "__main__":
    # 读取配置
    conf = configparser.ConfigParser()

    # 设置读入格式
    conf.read('./fileConfig', encoding='utf8')
    orgData = conf.get("orgFile", "orgdata")  # 原始数据路径
    newdicFile = conf.get("orgFile", "idictfile")  # 自定义词库路径
    stopwordsFile = conf.get("orgFile", "stopwordsfile")  # 停用词路径
    processed_date = conf.get("orgFile", "processed_date")  # 处理后的数据存储路径
    tfidf_vecmodel = conf.get("model", "tfidf_vecmodel")  # tfidf词向量存储路径
    bow_vecmodel = conf.get("model", "bow_vecmodel")  # BOW词向量存储路径

    # 数据处理
    data_after_stop, labels = data_process(newdicFile, stopwordsFile, orgData)
    data_after_stop["labels"] = labels  # 标签
    data_after_stop.to_excel(processed_date, index=False)  # 存储处理好的数据

    # 文本向量化
    textVectorisers = text2vector(data_after_stop, tfidf_vecmodel, bow_vecmodel)
