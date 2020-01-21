import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from data_handler import Data_Handler

class Tfidf_Experiment(object):
    """
    Description:
    ------------
    Uses Glove for text embedding on a labeled corpus. Then applies an ANN to
    predict labels on test data. 

    Parameters:
    -----------
    embedding_dim : ~int
        Dimension of vectors in the space in which words are represented.
        Accepts 100, 200 or 300. Default is 100.    
    
    Useful resources:
    -----------------
    https://www.kaggle.com/sermakarevich/sklearn-pipelines-tutorial
    
    Return:
    -------
    None
    """        
    def __init__(self, path_to_corpus):
        self.path_to_corpus = path_to_corpus

        # default params
        self.seed = 234012587
        self.test_split_fraction = .2
        
        self.df = None
        self.Train_X, self.Test_X = None, None
        self.Train_y, self.Test_y = None, None
        self.p = None
        
        self.run_experiment()

    @property
    def path_to_corpus(self):
        return self._path_to_corpus
    
    @path_to_corpus.setter
    def path_to_corpus(self, value):
        if type(value) is not str:
            raise ValueError('"path_to_corpus" must be a string!')
        self._path_to_corpus = value
        
    def collect_data(self):
        self.df = pd.read_csv(self.path_to_corpus, encoding='latin-1')
        self.df.dropna(how='all', inplace=True)

    def split_data(self):
        X = self.df.text
        y = self.df.label
        self.Train_X, self.Test_X, self.Train_y, self.Test_y = train_test_split(
          X, y, test_size=self.test_split_fraction, random_state=self.seed)

        self.max_length = max([len(s.split()) for s in self.Train_X])

    def encode_target(self):
        Encoder = LabelEncoder()
        self.Train_y = Encoder.fit_transform(self.Train_y)
        self.Test_y = Encoder.fit_transform(self.Test_y)        

    def perform_embedding(self):
        DH = Data_Handler()
        self.Train_X = DH.clean_tokenize(self.Train_X)
        self.Test_X = DH.clean_tokenize(self.Test_X)

        
    def create_pipeline(self):
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        SVM = svm.SVC(kernel='linear', degree=3, gamma='auto')      
        
        self.p = Pipeline([('tfidf', Tfidf_vect),
                           ('svm', SVM)
                           ])
        
    def train_model(self):

        param_grid = {'svm__C': np.logspace(-2., 1., 10)}       

        grid = GridSearchCV(
          self.p, cv=4, n_jobs=4, param_grid=param_grid, scoring='roc_auc', 
          return_train_score=True, verbose=0)
        grid.fit(self.Train_X, self.Train_y)
                
        predictions = grid.predict(self.Test_X)
        print('Model accuracy',accuracy_score(predictions, self.Test_y)*100.)
    
    def run_experiment(self):
        self.collect_data()
        self.split_data()
        self.encode_target()
        self.perform_embedding()
        self.create_pipeline()
        self.train_model()

if __name__ == '__main__':
    Tfidf_Experiment(path_to_corpus='./../data/corpus.csv')
