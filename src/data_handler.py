from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Data_Handler(object):
    """
    Description:
    ------------
    Collection of functions to prepare the data for NLP applications,
    especifically where TFIDF is used for word-embedding and NLTK is used for
    POS (part-of-speech) tagging.
    
    Parameters:
    -----------
    None needed.

    Source:
    -------
    The function nltk2wordnet below was adapted from
    https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
    
    Return:
    -------
    Wordnet object tag.
    """        
    def __init__(self):
        pass

    def nltk2wordnet(self, nltk_tag):
        """Dictionary to map NLTK POS tags and word2vec tags. 
        """
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def clean_tokenize(self, text):
        """Tokenize and clean corpus.
        """
        #Convert all text to lower case.
        text = text.str.lower()

        #Create a tokenizer. Split long strings into words.
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = [tokenizer.tokenize(words) for words in text.values]

        #Remove non-alphabetic words.
        tokens = [[word for word in sentence if word.isalpha()]
                  for sentence in tokens]

        #Remove stop words, such as 'a', 'the', etc.
        stop_words = set(stopwords.words('english'))
        tokens = [[w for w in words if not w in stop_words]
                  for words in tokens]

        #POS tagging. Tag words as verbs, adverbs and so on.
        tokens = [pos_tag(words) for words in tokens]

        #Convert POS tag to wordnet tag. Needed for lemmatizer.
        tokens = [[(w[0],self.nltk2wordnet(w[1])) for w in words]
                  for words in tokens]

        #Lemmatize words. I.e., remove conjugation, etc. flowers --> flower.
        lemmatizer = WordNetLemmatizer()
        tokens = [[lemmatizer.lemmatize(w[0], w[1]) for w in words
                  if w[1] is not None] for words in tokens]

        #Assemble lists back to strings to use in TFIDF.
        tokens_clean = [' '.join(words) for words in tokens]

        return tokens_clean
