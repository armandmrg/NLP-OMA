import numpy as np
import pandas as pd
import word2vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.linear_model import LogisticRegression


class Classifier:
    """The Classifier"""

    #############################################     
    def remove_stopwords(self,sentence):
        stop_words = set(stopwords.words('english')) 
        stop_words.remove("not")
        stop_words.remove("no")
        sentence = sentence.lower()
        sentence_tok = word_tokenize(sentence)
        sentence_f = ""
        for i in range(len(sentence_tok)):
            w=sentence_tok[i]
            if w not in stop_words:
                if i==len(sentence_tok)-1:
                    sentence_f+=w
                else:
                    sentence_f+=w+" "
        if len(sentence_f)<2:
            sentence_f = sentence
        return sentence_f

    def read_data(self,source):
        df=pd.read_csv(source,sep='\t',header=None)
        df.columns=["polarity","aspect_category","target_term","character_offset","sentence"]
        df["label"]=df["polarity"].apply(lambda x: 1 if x=="positive" else (0 if x=="neutral" else -1))
        #remove target
        sentence_red=[0]*len(df)
        for i in range(len(df)):
            sentence_red[i]=df["sentence"][i][:int(df["character_offset"][i].split(":")[0])]+df["sentence"][i][int(df["character_offset"][i].split(":")[1]):]
        df["sentence_red"]=sentence_red
        #remove stopwords
        df["sentence_red"]=df["sentence_red"].apply(lambda x:self.remove_stopwords(x))
        #word2vec embeddings
        PATH_TO_DATA = Path('C:/Users/Armand/Desktop/3A/Deep Learning/nlp_project/nlp_project/')
        en_embeddings_path = PATH_TO_DATA / 'cc.en.300.vec.gz'
        if not en_embeddings_path.exists():
            urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz', en_embeddings_path)
        w2vec = word2vec.Word2vec(en_embeddings_path, vocab_size=50000)
        sentence2vec = word2vec.BagOfWords(w2vec)
        sentences_emb=[sentence2vec.encode(df["sentence_red"][i]) for i in range(len(df["sentence_red"]))]
        return(sentences_emb,df["label"])
        
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        sentences_emb,labels=self.read_data(trainfile)
        logReg = LogisticRegression(penalty="l2",C = 10, multi_class='auto',solver='newton-cg')
        logReg.fit(sentences_emb,labels)
        self.clf=logReg
        
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        sentences_emb,labels=self.read_data(datafile)
        predictions=self.clf.predict(sentences_emb)
        polarity=[]
        for p in predictions:
            if p==1:
                polarity.append("positive")
            elif p==0:
                polarity.append("neutral")
            else:
                polarity.append("negative")
        return (polarity)





