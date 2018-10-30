import os
from gensim.models.wrappers import LdaMallet
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from DocumentRetriever import DocumentRetriever

class LdaMalletHandler:
    def __init__(self, mallet_path):
        self.mallet_path = mallet_path

    def run_model(self, model_name, corpus, num_topics, workers=8):
        self.model_name = model_name
        self.dictionary = Dictionary(corpus)
        corpus_bow = [self.dictionary.doc2bow(text) for text in corpus]
        os.makedirs("models/"+model_name, exist_ok=True )
        self.model = LdaMallet(self.mallet_path, corpus_bow, num_topics=num_topics,id2word=self.dictionary, workers=workers, prefix="./models/"+model_name+"/")

    def save_model(self):
        self.model.save("models/"+self.model_name+"/model.model")
        self.dictionary.save("models/"+self.model_name+"/dict.dict")

    def load_model(self, model_name):
        self.model_name = model_name
        self.dictionary  = corpora.Dictionary.load("models/"+self.model_name+"/dict.dict")
        self.model = LdaMallet.load("models/"+self.model_name+"/model.model") 

    def ext_doc_topics(self, ext_doc):
        doc_bow = self.dictionary.doc2bow(ext_doc)
        doc_topics = self.model[doc_bow]
        doc_topics.sort(key=lambda x: x[1], reverse=True)
        return doc_topics

    def ext_doc_n_most_similar(self, ext_doc, n=5, metric='cosine'):
        if(not hasattr(self, 'doc_retriever')):
            self.doc_retriever =  DocumentRetriever(self.model.fdoctopics())
        doc_bow = self.dictionary.doc2bow(ext_doc)
        doc_topics = self.model[doc_bow]
        topics = []
        for topic in doc_topics:
            topics.append(topic[1])    
        most_similar = self.doc_retriever.n_most_similar(topics, n=n, metric=metric)    
        return most_similar