#!/bin/bash
#-*-coding:utf-8-*-
#encoding:utf-8
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
import jieba
import numpy
from sklearn.datasets.base import Bunch
import cPickle as pickle



"""
初始语料库的预处理工作，为后期方便调用做准备。。
"""
class Textprocess:
    data_set = Bunch(content=[],label=[],person = [])
    # contents:文件内容,包含content和title，整合在一起了。
    # label:每个文件的分类标签列表
    # filenames：人名列表
    word_weight_bag = Bunch(tdm=[], vocabulary=[],label=[],person = [])
    #tdm 是语料库权重矩阵对象
    

    def __init__(self):
        #语料库的原始路径
        self.corpus_path = ""
        #预处理后的语料库路径
        self.pos_path = ""
        #分词的语料库的持久化数据结构和IF-IDF数据结构的存储文件路径
        self.wordbag_path =""
        #停用词路径
        self.stopword_path =""
        #训练集和测试集的set存储的文件名
        self.trainset_name =""
        #文本的If-idf:word_weight_bag的文件名
        self.word_weight_bag_name =""


    """
    功能：原始预料库中的文本，经过分词以及简单的预处理操作，并将处理后的文本存在 pos_path 目录下
    参数：corpus_path：原始原料库的路径
          pos_path ：预处理后的语料库的路径
    """
    def corpus_segment_load_pos_path(self):
        if (self.corpus_path =="" or self.pos_path == ""):
            print "原始语料库的路径（corpus_path）or 预处理语料库的存储路径（pos_path）不存在。"
            return

        #获取原始语料库目录下的所有文件的列表
        dir_list = os.listdir(self.corpus_path)
        for mydir in dir_list:
            file_path = os.path.exists(self.corpus_path,mydir)
            #print file_path
            person = mydir[:-4]
            print person.encode('utf-8')
            with open(file_path,'r') as f:
                with open(os.path.exists(pos_path,mydir),'w') as g:
                    for line in f.readlines():
                        line = line.strip()
                        line = ' '.join(jieba.cut(line,cut_all=False))
                        g.write('%s\n' % line) 

 
    #匹配每一行（一个文本）中文
    def match_chinese(self,line):
        xx=u"[\u4e00-\u9fa5]+"
        pattern = re.compile(xx)
        line = ' '.join(pattern.findall(line))
        return line


    """
    功能：根据听停用词文本的路径：stopword_path 得到停用词的列表
    返回值：stopword_list 为停用词列表
    """
    def getstopword(self,stopword_path):
        stop_file =open(stopword_path,'rb')
        stop_content = stop_file.read()
        stopword_list = stop_content.splitlines()
        stop_file.close()
        return stopword_list

    
    """
    功能：将分词处理后的语料库和对应的标记，持久化于data_set数据结构中，存储于wordbag_path文件中的，命名为trainset_name
    参数：实际没有，但是根据需要必须提供：
          wordbag_path：持久化的文件路径
          pos_path：预处理的语料库路径
          trainset_name：训练集或者测试集的文件名
    """
    def pos_path_load_data_set(self):
        if self.wordbag_path == "" or self.pos_path == ""  or self.trainset_name == "":
            print "分词后持久化存储的路径（wordbag_path）和分词后原始文件路径（pos_path）都不能为空,同时持久化存储的文件名也不能为空。"
            return

        #读取所有人名的标记字典
        with open(os.path.join(os.path.dirname(os.getcwd()),'target_all'),'r') as f:
            target_all = pickle.loads(f.read())

        dir_list = os.listdir(self.pos_path)
        for mydir in dir_list:
            person_name = mydir[:-4]
            file_path_list = os.path.join(self.pos_path,mydir)
            with open(file_path_list,'r') as f:
                seg_corpus = f.readlines()
            self.data_set.content.append(seg_corpus)
            self.data_set.person.append(person_name)
            self.data_set.label.append(target_all[person])

        #将data_set对象存储于文件wordbag_path + trainset_name中
        if not os.path.exists(self.wordbag_path):
            os.makedirs(self.wordbag_path)
        file_obj = open(self.wordbag_path + self.trainset_name,'wb')
        pickle.dump(self.data_set,file_obj)
        file_obj.close()
        print "分词后语料库打包到data_set中成功，并保存在文件wordbag_path/trainset_name中。"
        print "##########################################################################################"


    """
    功能：将date_set中的数据，通过TF-idf计算其，文本的向量空间模型，并将具体具体的值保存在wordbag_path目录下的word_weight_bag_name文件名中
    参数：wordbag_path：持久化的文件路径 也是TD-IDF 矩阵的存储路径
          word_weight_bag_name: TF-IDF矩阵存储的文件名
    """
    def if_idf_load_word_weight_bag(self):
        if self.wordbag_path == "" or word_weight_bag_name == "":
            print "wordbag_path or word_weight_bag_name is None."
            return
        for index,person in enumerate(self.data_set.person):
            corpus = self.data_set.content[index]
            vectorizer = TfidfVectorizer(max_features=1000, use_idf=True)
            weight_matrix = vectorizer.fit_transform(corpus).todense()
            vocabulary_dict = vectorizer.vocabulary_
            self.word_weight_bag.tdm.append(weight_matrix)
            self.word_weight_bag.vocabulary.append(vocabulary_dict)
            self.word_weight_bag.label.append(self.data_set.label[index])
            self.word_weight_bag.person.append(person)

        if not os.path.exists(wordbag_path):
            os.makedirs(wordbag_path)
        wordbag_obj = open(os.path.exists(wordbag_path,word_weight_bag_name),'wb')
        pickle.dump(self.word_weight_bag,wordbag_obj)
        wordbag_obj.close()

        print "语料库的TF-IDF的矩阵计算完毕！"
        print "==================================================================="
    
    """
    功能：如果先前已经将TF-IDF值训练好了，可以直接导出data_set
    返回值：data_set数据
    """
    def load_trainset(self):
        file_obj =open(self.wordbag_path + self.trainset_name,'rb')
        self.data_set = pickle.load(file_obj)
        file_obj.close()
        return self.data_set
    """ 
    功能：如果先前已经将TF-IDF值训练好了，可以直接导出word_weight_bag
    返回值：为word_weight_bag数据
    """
    def load_word_weight_bag(self):
        file_obj = open(self.wordbag_path +self.word_weight_bag_name,'rb')
        self.word_weight_bag =pickle.load(file_obj)
        file_obj.close()
        return self.word_weight_bag


"""
'''
计算title和content的向量空间矩阵，并通过线性的权重参数组合成文本的VSM矩阵
file_path_title:title文件的路径
file_path_content:content文件的路径
a,b:分别为title和content的权重
'''
def VSM(file_path_title,file_path_content,a,b):
    corpus1 = []
    corpus2 = []
    for line in open(file_path_title, 'r').readlines():  
        corpus1.append(line.strip())

    for line in open(file_path_content, 'r').readlines():  
        corpus2.append(line.strip())

    vectorizer1 = TfidfVectorizer(max_features=100, use_idf=True)
    vectorizer2 = TfidfVectorizer(max_features=1000, use_idf=True)
    weight_title = vectorizer1.fit_transform(corpus1).todense()
    weight_content = vectorizer2.fit_transform(corpus2).todense()
    #根据组合形成最终的VSM向量空间
    weight = numpy.hstack((weight_title*a,weight_content*b))
    #输出提取的特征字典
    #word_list_title = vectorizer1.vocabulary_ 
    #word_list_content = vectorizer2.vocabulary_    
    return weight_title,weight_content,weight
"""

    

if __name__ == '__main__':
    
    
 

