import numpy as np
import pandas as pd
import nltk

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from data_handler import Data_Handler

#Uncomment the following to install nltk subpackages. Execute only once.
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

class Tfidf_Experiment(object):
    """
    Description:
    ------------
    Uses TFIDF for text embedding on a labeled corpus. Then applies an SVM to
    predict labels on test data. 

    Parameters:
    -----------
    path_to_corpus : ~str
        Relative or absolute path to the file corpus.csv.    
    
    Useful resources:
    -----------------
    https://www.kaggle.com/sermakarevich/sklearn-pipelines-tutorial
    
    Return:
    -------
    None
    """        
    def __init__(self, path_to_corpus):
        self._path_to_corpus = path_to_corpus

        # default params
        self.seed = 234012587
        self.test_split_fraction = .2
        
        self.df = None
        self.p = None
        self.Train_X, self.Test_X = None, None
        self.Train_y, self.Test_y = None, None
        
        self.run_experiment()

    @property #Property enforcement of input variables.
    def path_to_corpus(self):
        return self._path_to_corpus
    
    @path_to_corpus.setter
    def path_to_corpus(self, value):
        if type(value) is not str:
            raise ValueError('"path_to_corpus" must be a string!')
        self._path_to_corpus = value
        
    def collect_data(self):
        """Read data from corpus and remove nan rows.
        """
        self.df = pd.read_csv(self.path_to_corpus, encoding='latin-1')
        self.df.dropna(how='all', inplace=True)

    def split_data(self):
        """Divide the data into train and test. Train data will later be futher
        divided in validation subsets. 
        """
        X = self.df.text
        y = self.df.label
        self.Train_X, self.Test_X, self.Train_y, self.Test_y = train_test_split(
          X, y, test_size=self.test_split_fraction, random_state=self.seed)

    def encode_target(self):
        """Encode labels. Since this is a binary classification problem, the
        labels will be assigned either 0 or 1.
        """
        Encoder = LabelEncoder()
        self.Train_y = Encoder.fit_transform(self.Train_y)
        self.Test_y = Encoder.fit_transform(self.Test_y)        

    def perform_embedding(self):
        """Use an instance of the Data_Handler class to tokenize the data.
        POS-tagging is enforced.
        """
        DH = Data_Handler()
        self.Train_X = DH.clean_tokenize(self.Train_X)
        self.Test_X = DH.clean_tokenize(self.Test_X)
        
    def create_pipeline(self):
        """Create a 2-step pipeline for the ML application. The data processing
        could be included here as well.
        """
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        SVM = svm.SVC(kernel='linear', degree=3, gamma='auto')      
        
        self.p = Pipeline([('tfidf', Tfidf_vect),
                           ('svm', SVM)
                           ])
        
    def train_model(self):
        """Train model using a parameter grid to optimize the regularization
        parameter 'C' in the SVM model.
        """
        param_grid = {'svm__C': np.logspace(-2., 1., 8)}       

        grid = GridSearchCV(
          self.p, cv=4, n_jobs=4, param_grid=param_grid, scoring='roc_auc', 
          return_train_score=True, verbose=0)
        grid.fit(self.Train_X, self.Train_y)
                
        #Make predictions using the fitted model and the test data.
        predictions = grid.predict(self.Test_X)
        print('Model accuracy',accuracy_score(predictions, self.Test_y)*100.)
    
    def run_experiment(self):
        """Call all the routines above.
        """
        self.collect_data()
        self.split_data()
        self.encode_target()
        self.perform_embedding()
        self.create_pipeline()
        self.train_model()

if __name__ == '__main__':
    Tfidf_Experiment(path_to_corpus='./../data/corpus.csv')
