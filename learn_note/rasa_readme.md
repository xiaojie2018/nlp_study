
# RARA

- [基于RASA的task-orient对话系统解析（一）](https://zhuanlan.zhihu.com/p/75517803)
- [基于RASA的task-orient对话系统解析（二）——对话管理核心模块](https://zhuanlan.zhihu.com/p/78665885)
- [基于RASA的task-orient对话系统解析（三）——基于rasa的会议室预定对话系统实例](https://zhuanlan.zhihu.com/p/81430436)
- [参考资料1](https://blog.csdn.net/ljp1919/article/details/103975263)


## RASA_NUL

### 1. 词向量
#### 1.1 MitieNLP
#### 1.2 SpacyNLP
### 2. 文本特征化
#### 2.1 MitieFeaturizer
#### 2.2 SpacyFeaturizer
#### 2.3 ConveRTFeaturizer
#### 2.4 RegexFeaturizer
#### 2.5 CountVectorsFeaturizer
### 3. 意图分类器
#### 3.1 MitieIntentClassifier
#### 3.2 SklearnIntentClassifier
#### 3.3 EmbeddingIntentClassifier
#### 3.4 KeywordIntentClassifier
### 4. 选择器Selectors
### 5. 分词器Tokenizers
#### 5.1 WhitespaceTokenizer
#### 5.2 JiebaTokenizer
#### 5.3 MitieTokenizer
#### 5.4 SpacyTokenizer
### 6. 实体抽取器Entity Extractors
#### 6.1 MitieEntityExtractor
#### 6.2 SpacyEntityExtractor
#### 6.3 EntitySynonymMapper
#### 6.4 CRFEntityExtractor
#### 6.5 DucklingHTTPExtractor


# 数据格式

- what is my balance <!-- no entity -->
- how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" -->
- how much do I have on my [savings account](source_account:savings) <!-- synonyms, method 1-->
- Could I pay in [yen](currency)?  <!-- entity matched by lookup table -->

## intent:greet
- hey
- hello

## synonym:savings   <!-- synonyms, method 2 -->
- pink pig

## regex:zipcode
- [0-9]{5}

## lookup:additional_currencies  <!-- specify lookup tables in an external file -->
- path/to/currencies.txt

'''
    说明：
        Rasa NLU 数据集可以结构化为4个部分：
            常见示例
            同义词
            正则化特征
            查找表

'''
## Json格式
- 由top level对象rasa_nlu_data所包含，对应的keys有：common_examples(最为重要), entity_synonyms和regex_features。具体示例如下：
{
    "rasa_nlu_data": {
        "common_examples": [],
        "regex_features" : [],
        "lookup_tables"  : [],
        "entity_synonyms": []
    }
}





### 输入
