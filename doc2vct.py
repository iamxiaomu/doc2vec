#coding:utf-8
#使用doc2vec 判断文档相似性
from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os

def a_sub_b(a,b):
    ret = []
    for el in a:
        if el not in b:
            ret.append(el)
    return ret
stop = [line.strip().decode('utf-8') for line in open('stopword.txt').readlines() ]

#读取文件
raw_documents=[]
walk = os.walk(os.path.realpath("/Users/muhongfen/sougou"))
for root, dirs, files in walk:
    for name in files:
        f = open(os.path.join(root, name), 'r')
    raw = str(os.path.join(root, name))+" "
    raw += f.read()
    raw_documents.append(raw)

#构建语料库
corpora_documents = []
doc=[]            #输出时使用，用来存储未经过TaggedDocument处理的数据，如果输出document，前面会有u
for i, item_text in enumerate(raw_documents):
	words_list=[]
	item=(pseg.cut(item_text))
	for j in list(item):
		words_list.append(j.word)
	words_list=a_sub_b(words_list,list(stop))
	document = TaggedDocument(words=words_list, tags=[i])
	corpora_documents.append(document)
	doc.append(words_list)
#创建model
model = Doc2Vec(size=50, min_count=1, iter=10)
model.build_vocab(corpora_documents)
model.train(corpora_documents)
print('#########', model.vector_size)

#训练
test_data_1 = '本报讯 全球最大个人电脑制造商戴尔公司８日说，由于市场竞争激烈，以及定价策略不当，该公司今年第一季度盈利预计有所下降。'\
'消息发布之后，戴尔股价一度下跌近６％，创下一年来的新低。戴尔公司估计，其第一季度收入约为１４２亿美元，每股收益３３美分。此前公司预测当季收入为１４２亿至１４６亿美元，'\
'每股收益３６至３８美分，而分析师平均预测戴尔同期收入为１４５．２亿美元，每股收益３８美分。为抢夺失去的市场份额，戴尔公司一些产品打折力度很大。戴尔公司首席执行官凯文·罗林斯在一份声明中说，'\
'公司在售后服务和产品质量方面一直在投资，同时不断下调价格。戴尔公司将于５月１８日公布第一季度的财报。'
test_cut_raw_1 =[]
item2=(pseg.cut(test_data_1))
for k in list(item2):
	test_cut_raw_1.append(k.word)
inferred_vector = model.infer_vector(test_cut_raw_1)
sims = model.docvecs.most_similar([inferred_vector], topn=3)
print(sims)  #sims是一个tuples,(index_of_document, similarity)
for i in sims:
	similar=""
	print('################################')
	print i[0]
	for j in doc[i[0]]:
		similar+=j
	print similar