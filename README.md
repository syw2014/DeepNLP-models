# DeepNLP-Models

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE) 


Table of Contents
=================
1. [Introduction](#Introduction)
2. [NLP Tasks](#tasks)
    - [Text Matching](#text_match)
    - [Text Search](#text_search)
    - [Document Classification](#doc_classify)

3. [Learning to Rank](#rank)
    - [DeepFM](#deepfm)

## Introduction
As we all know deep learning has shown super power in image processing, so a lot researchers try to study NLP tasks with deep learning, and a various of 
deep nlp models haven discussion, that's what I want to do to collect the deep nlp models which can get the state-of-art performance, so that we can use
those models in industry.

## NLP Tasks <a name="tasks"></a>
This part will introduce useful deep learning models in nlp tasks, and we try to make it available in industry. And the models will contain basic tasks
like classification, text similarity matching, named entity recognition, ctr and some higher models(seq2seq) 
### Text Matching <a name="text_match"></a>
Text matching are wide used in infromation retrieval, question answering, conversational response ranking etc, so we choose some simple and easy
models, and implement them in an enffiency way with tensorflow. As a start, I will implement a baseline with [DSSM](https://www.microsoft.com/en-us/research/project/dssm/)  
and the introduction about this model in [here](https://github.com/syw2014/DeepNLP-models/blob/master/docs/dssm.md).

### Text Search <a name="text_search"></a>
Here we try to complete this task with another approach called $Deep Search$. In the current version we havn't add complicated algorithms, we just use word embedding and [Faiss](https://github.com/facebookresearch/faiss.git), embedding to encode query and answers, and faiss speed search related answers. This may also can be a simple way to do QA, but the result may not very good.Check [here](https://github.com/syw2014/DeepNLP-models/blob/master/codes/TextSearch/deep_search.py) to find module.

- [x] Add implementation ideas of this parts
- [x] Collect more QA data 
- [x] Implement service for Query search.

### Document Classification <a name="doc_classify"></a>
    - [x] Add Descriptions
    - [x] Add implementation

## Learning To Rank <a name="rank"></a>
Learning to rank is an important task in recommendation, here I collect the most common models and algorithms to reimplement them with real dataset.
The different between the implementation with other repository in github was my implemenation can be used in your product project directly after 
modified the input pipeline proprecess.

### DeepFM <a name="deepfm"></a>
    - [x] Add description
    - [x] Add impemenation




## Contact
If you have any ideas or suggestion contact me with ***jerryshi0110@gmail.com***
