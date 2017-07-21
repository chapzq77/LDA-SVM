# encoding:utf-8
import jieba
import jieba.analyse
#import numpy
#import re
#import sys
from lxml import etree
import codecs
import os
#import chardet
import numpy as np
from gensim import corpora, models
import gensim
from six import iteritems
#from pprint import pprint
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import cPickle as pickle
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
import json

def file_file():
    sougou_path = os.getcwd().replace('\\','/') + '/' + 'tag'
    dir_list = os.listdir(sougou_path)
    file_all_path_list_preprocess = []
    target_all = {}
    person_list = []
    for dirs in dir_list:
        person = dirs[:-4]
        file_all_path = sougou_path + "/"+ dirs
        preprocess_file_path_name,target = read_every_file_extract(file_all_path,person)
        #所有人名的标记，每个人名对应一个target列表
        target_all.setdefault(person,target)
        person_list.append(person)
        file_all_path_list_preprocess.append(preprocess_file_path_name)
        """
        存储label的字典，在target_all文件中
        """
        label = pickle.dumps(target_all)
        with open('E:/pythonUse/experimental_procedure/LDA-SVM/target_all','w') as f:
            g.write(label)
            
    return target_all,person_list,file_all_path_list_preprocess
  

"""
读取文件，并且抽取文件中的内容，构造预处理后的文本：
参数：file_path：原始文件路径
      person：路径对应的人名
return：preprocess_file_path_name：预处理后的文件路径
        target：该person 对应的标签

"""
def read_every_file_extract(file_path,person):
    preprocess_file_path = os.getcwd().replace('\\','/') + "/" + 'preprocess'
    #以utf8的格式读取文件的内容
    if not os.path.exists(preprocess_file_path):
        os.makedirs(preprocess_file_path)
    preprocess_file_path_name = preprocess_file_path + "/" + str(person) + ".txt"
    try:
        html = codecs.open(file_path, 'r', 'utf-8').read()
    except:
        html = codecs.open(file_path, 'r').read()

    root = etree.HTML(html.encode('utf-8').decode('utf8','ignore'))
    #xx=u"[\u4e00-\u9fa5]+"
    #pattern = re.compile(xx)
    content = []
    for ele in root.xpath('//doc'):
        content1 = (''.join(ele.xpath('./content//text()')))
        tag = ele.xpath('./tag/text()')[0]
        title = (''.join(ele.xpath('./contenttitle//text()')))
        content.append({'tag':tag,'title':title,'content1':content1})
    target = []
    f = open(preprocess_file_path_name,'wb')

    for i in range(len(content)):
        s1 = str(content[i]['title'].encode('utf8'))
        s2 = str(content[i]['content1'].encode('utf8'))
        target.append(int(content[i]['tag']))
        f.write('%s%s\n' % (s1,s2))
    f.close()
    return preprocess_file_path_name,target


#获得停用词列表，输入参数：停用词文件的路径:stopwords_path
def get_stop_word(stopwords_path):
    stopword_list = []
    f = open(stopwords_path,'r')
    for i in f.readlines():
        stopword_list.append(i.split())
    return stopword_list



"""
LDA的文本预处理，将文本经过过滤，形成字典，并将文本形成corpus的形式，存储在tmp文件中
"""    
def LDA_pre_text(stopwords_path):
    target_all,person_list,file_all_path_list_preprocess = file_file()
    person = person_list[0]
    #print person.decode('gbk').encode('utf-8')
    preprocess_file_path_name = file_all_path_list_preprocess[0]
    #print preprocess_file_path_name.decode('gbk').encode('utf-8')
    target = target_all[person]
    #print len(target)
    # 读取文件中的文本，每行一个文本。
    f = open(preprocess_file_path_name, 'r')
    documents = f.readlines()
    #print preprocess_file_path_name
    #print len(documents)

    # 构造文本和词的分词统计texts矩阵
    texts = [[word for word in jieba.cut(document, cut_all=False)] for document in documents]
    #print len(texts)
    # pprint(texts)
    dictionary = corpora.Dictionary(texts)
    #得到停用词列表
    stopword_list = get_stop_word(stopwords_path)
    stop_ids = [dictionary.token2id[stopword[0]] for stopword in stopword_list if stopword[0] in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq <= 2]
    #移除停用词和文本中只出现一次的词
    dictionary.filter_tokens(stop_ids+once_ids)
    #删除字典移除后的空白id序列
    dictionary.compactify()
    model_path = "E:/pythonUse/experimental_procedure/LDA-SVM/tmp"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    dirctionary_path = model_path + "/dict.dict"
    dictionary.save(dirctionary_path)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(model_path + '/corpus.mm', corpus)

    """
    for doc in corpus:
        print doc
    """
    #将标记转化为矩阵（也就是向量的形式）
    target = np.array(target)
    #print type(target)
    #print target
    label = pickle.dumps(target)
    g = open('E:/pythonUse/experimental_procedure/LDA-SVM/pickle','w')
    g.write(label)
    g.close()


  
"""
装载语料库和字典，使用LDA进行降维，得到文本*主题矩阵,并且存储（主题*文本矩阵）以及（target）
"""  
def LDA():
    if os.path.exists("E:/pythonUse/experimental_procedure/LDA-SVM/tmp"):
        dictionary =corpora.Dictionary.load("E:/pythonUse/experimental_procedure/LDA-SVM/tmp/dict.dict")
        corpus = corpora.MmCorpus('E:/pythonUse/experimental_procedure/LDA-SVM/tmp/corpus.mm')
        print "装载结束!"
    else:
        print "原来没有语料库,请完成字典和语料库的操作."
    #存储主题的路径
    topic_matrix_path = 'E:/pythonUse/experimental_procedure/LDA-SVM/topic'
    if not os.path.exists(topic_matrix_path):
        os.makedirs(topic_matrix_path)
    
    for topic_num in range(10,500,5):

        model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_num)
        corpus_topic = model[corpus]
        #得到的矩阵的每列代表一个文本中主题的分布
        numpy_matrix = gensim.matutils.corpus2dense(corpus_topic,num_terms = topic_num)
        #矩阵的转置
        numpy_matrix = numpy_matrix.T
        """
        package_topic_matrix = {}
        package_topic_matrix['topic_num']= topic_num
        package_topic_matrix['matrix'] = numpy_matrix
        """
        #存储每个矩阵和主题的的路径
        topic_path = topic_matrix_path + "/" +str(topic_num)
        label = pickle.dumps(numpy_matrix)
        g = open(topic_path,'w')
        g.write(label)
        g.close()


        #print numpy_matrix.shape
        #print numpy_matrix
        

        #print type(target)
        #print target
        #numpy_matrix = np.transpose(numpy_matrix.shape)


def SVM(topic_matrix_path):
    # target
    f = open('E:/pythonUse/experimental_procedure/LDA-SVM/pickle','r')
    target = f.read()
    f.close()
    target = pickle.loads(target)
    #存储最好的参数结果！！！
    with open("best_params.json",'w') as h:
        for topic_num in range(10,480,5):
            kf = KFold(n_splits=2)
            g = open(os.path.join(topic_matrix_path,str(topic_num)),'r')
            numpy_matrix = g.read()
            numpy_matrix = pickle.loads(numpy_matrix)
            g.close()
            for train_index, test_index in kf.split(numpy_matrix):
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = numpy_matrix[train_index], numpy_matrix[test_index]
                y_train, y_test = target[train_index], target[test_index]

                C_range = range(1,11,1)
                #gamma_list=[0.125, 0.25, 0.5 ,1, 2, 4]
                penalty_range=['l1','l2']
                param_grid=dict(C=C_range,penalty=penalty_range)
                SVM = LinearSVC(C=1,penalty="l1",dual=False,tol=1e-4)
                grid = GridSearchCV(SVM,param_grid,cv=2)
                grid.fit(X_train,y_train)
                cv_result = pd.DataFrame.from_dict(grid.cv_results_)
                param_result_path = "E:/pythonUse/experimental_procedure/LDA-SVM/param_result"
                if not os.path.exists(param_result_path):
                    os.makedirs(param_result_path)
                
                with open(os.path.join(param_result_path,str(topic_num)),'w+') as f:
                    cv_result.to_csv(f)
                #print grid.grid_scores_
                #print type(grid.best_params_)
                best_params = json.dumps([grid.best_params_,topic_num])
                h.write(best_params)
                #得到最好的估计值模型
                bclf = grid.best_estimator_  
                #训练模型
                bclf.fit(X_train, y_train) 
                y_pred = bclf.predict(X_test) 
                print "++++++++++++++++++++the number of topic are %d" %topic_num
                print classification_report(y_test, y_pred)

            #print grid.best_params_
"""
def kmeans_method():

"""


    

if __name__ == '__main__':
    topic_matrix_path = 'E:/pythonUse/experimental_procedure/LDA-SVM/topic'

    #stopwords_path = "E:/pythonUse/experimental_procedure/LDA-SVM/stopword/ch_stop_words.txt"   
    #LDA_pre_text(stopwords_path)
    SVM(topic_matrix_path)
    #target_all,person,preprocess_file_path_name = file_file()
    #print target_all[person[0]],person[0]
    #file_file()