# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/10 17:56
# software: PyCharm



corpus = [
  "帮我 查下 明天 北京 天气 怎么样",
  "帮我 查下 今天 北京 天气 好不好",
  "帮我 查询 去 北京 的 火车",
  "帮我 查看 到 上海 的 火车",
  "帮我 查看 特朗普 的 新闻",
  "帮我 看看 有没有 北京 的 新闻",
  "帮我 搜索 上海 有 什么 好玩的",
  "帮我 找找 上海 东方明珠 在哪"
]
from sklearn.feature_extraction.text import CountVectorizer
# step 1
vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
# step 2
vectoerizer.fit(corpus)
# step 3
bag_of_words = vectoerizer.get_feature_names()
print("Bag of words:")
print(bag_of_words)
print(len(bag_of_words))
# step 4
X = vectoerizer.transform(corpus)
print("Vectorized corpus:")
print(X.toarray())
# step 5
print("index of `的` is : {}".format(vectoerizer.vocabulary_.get('的')))


from sklearn.feature_extraction.text import TfidfTransformer
# step 1
tfidf_transformer = TfidfTransformer()
# step 2
tfidf_transformer.fit(X.toarray())
# step 3
for idx, word in enumerate(vectoerizer.get_feature_names()):
    print("{}\t{}".format(word, tfidf_transformer.idf_[idx]))
# step 4
tfidf = tfidf_transformer.transform(X)
print(tfidf.toarray())


