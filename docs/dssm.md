# Deep Structured Semantic Models
该模型主要目的是计算query和document的相似度，而[论文](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)提到的搜索场景下，用户输入一个query，该怎么返回召回的documents，所以建立这个模型来计算query和doc直接的相似度。


## 模型特点：
- 数据构造
在搜索场景下，利用query对应点击过的doc 和未点击过的doc组成训练样本对（采用点击与否来表示相关性），进行训练模型
- 模型
模型采用一般的DNN方式，网络框图如下，
![image.png](https://upload-images.jianshu.io/upload_images/2423131-bb493d7df6bcf808.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

激活函数采用，$tanh$， loss计算如下：
![image.png](https://upload-images.jianshu.io/upload_images/2423131-29bc44d95b9655a0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

整个模型是首先通过对query 和doc中的term 进行编码， 然后做三层FC，最后接softmax层输出计算loss，softmax层的输入是经过三层FC后的query和doc的余弦距离，loss计算的是query和被点doc余弦距离的负对数。

## 模型存在的问题
- Term vector， 可以通过one-hot进行编码，由于vocabulary 会很大，且存在未登录词，模型进行了改进，采用***word hash***，即将单词拆分成如下形式，"#good#" -> "#go, goo, ood, od#"。但对于中文来说，此种方式不见效，中文可以拆分成单字，偏旁、部首，后者做embedding。


# 模型实现
TO BE Done
