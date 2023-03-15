# imports
import re
import nltk
import pickle

import numpy as np
import pandas as pd
import unicodedata
import contractions
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from autocorrect import Speller 
from nltk.stem import RSLPStemmer
from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

%matplotlib inline

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')

# funcoes de tratamento

def remove_stopwords(text, language):
    """
    text: text to be treated (str)
    language: chosen language for treatment of text (str)
    return: text without stopwords (str)
    """
    try:
        stop_words = set(stopwords.words(language)) 
        word_tokens = word_tokenize(text.lower())
        treated_text = ' '.join([w for w in word_tokens if not w in stop_words])
    except:
        return text
    return treated_text


def normalize_text (text):
    """
    text: text to be treated (str)
    return: text to normal form KD - all compatibility characters with their equivalents (str)
    """    
    try:
        treated_text = unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode() 
    except:
        return text
    return treated_text   


def remove_simple_noise(text):
    """
    text: text to be treated (str)
    return: text without multiple whitespace (str)
    """
    try:
        treated_text = re.sub('\s+', ' ', text)
        treated_text = re.sub(' +', ' ', treated_text)
        treated_text = re.sub('[\n\r]', ' ', treated_text)
        treated_text = re.sub(r'[^\w\s]+', ' ', treated_text) 
    except:
        return text
    return treated_text


def remove_punct_special_char(text):
    """
    text: text to be treated (str)
    return: text without punctuation and special characters (str)
    """
    try:
        treated_text = re.sub(r"[^a-zA-Z:$-,%.?!]+", ' ', text) 
        treated_text = re.sub(r'[^\w]', ' ', treated_text) 
        
    except:
        return text
    return treated_text


def expand_contractions(text):
    """
    text: text to be treated (str)
    return: standardized test without contractions (str)
    """
    try:
        treated_text = contractions.fix(text)
    except:
        return text
    return treated_text


def spelling_correction(text, language):
    """
    text: text to be analyzed (str)
    language: chosen language for correcting spellings (str)
    return: text without misspell (str)
    """
    spell = Speller(lang=language)
    corrected_text = spell(text)
    return corrected_text


##TODO melhorar o regex para pegar outros formatos e não só http/https
def find_hiperlinks(text):
    """
    text: text to be analyzed (str)
    return: text with hyperlinks replaced by name 'link' (str)
    """
    try:
        pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        new_text = re.sub(pattern, ' link ', text)
    except:
        return text
    return new_text


def find_dates(text):
    """
    text: text to be analyzed (str)
    return: text with date replaced by name 'date' (str)
    """
    pattern1 = '(\d{1,}\W\d{1,2}\W\d{1,4})'
    pattern2 = '(\d{1,}\W\d{1,2})'
    text = re.sub(pattern1, ' date ', text)
    text = re.sub(pattern2, ' date ', text)
    return text

def find_money(text):
    """
    text: text to be analyzed (str)
    return: text with money replaced by name 'dinheiro' (str)
    """
    try:
        pattern1 = '\w+\$[ ]{0,}\d+(,|\.)\d+'
        pattern2 = '\w+\$[ ]{0,}\d+'
        new_text = re.sub(pattern1, ' dinheiro ', text)
        new_text = re.sub(pattern2, ' dinheiro ', new_text)
    except:
        return text
    return new_text


def find_numbers(text):
    """
    text: text to be analyzed (str)
    return: text with numbers replaced by name 'numero' (str)
    """
    try:
        new_text = re.sub('[0-9]+', ' numero ', text)
    except:
        return text
    return new_text


def find_negation(text):
    """
    text: text to be analyzed (str)
    return: text with negation word replaced by name 'negação' (str)
    """
    try:
        new_text = re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', text)
    except:
        return text
    return new_text


#selecao "snowball" ou "rslp"
def stemming_process(text, language, stemmerName = "snowball"): 
    """
    text: text to be processed (str)
    language: chosen language for stemming process in case of snowball algorithm (str)
    stemmerName: name of stemmer. Default: snowball algorithm (str)
    return: text without misspell (str)
    """
    stemmer = nltk.stem.RSLPStemmer() if stemmerName == "rslp" else nltk.stem.snowball.SnowballStemmer(language) 
    text = [stemmer.stem(item) for item in text.split()]
    return " ".join(text)


def extract_features_from_corpus(corpus, vectorizer):
    """
    corpus: list of text to be transformed into a matrix (str)
    vectorizer: engine to be used in the transformation (object)
    return: text converted in vector (np array) and pandas df
    """
    try:    
        #Extracting features
        corpus_features = vectorizer.fit_transform(corpus).toarray()
        features_names = vectorizer.get_feature_names()
        
        #Transforming into a dataframe 
        df_corpus_features = pd.DataFrame(corpus_features, columns=features_names)
    except:
        return corpus, pd.DataFrame()
    return corpus_features, df_corpus_features


#Using CountVec
def ngrams_count(corpus, ngram_range, stopwords_language, n=-1):
    """
    corpus: text to be analysed (pd.DataFrame)
    ngram_range: type of n gram to be used on analysis (tuple)
    stopwords_language: chosen language for vectorize text (str)
    n: top limit of ngrams to be shown. default -1 (int)
    return: pandas dataframe with ngram and frequency
    """
    try:
        # Creating Bag of words
        vectorizer = CountVectorizer(stop_words=stopwords.words(stopwords_language), ngram_range=ngram_range).fit(corpus)
        bag_of_words = vectorizer.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        total_list = words_freq[:n]
        
        # Returning a DataFrame with the ngrams count
        count_df = pd.DataFrame(total_list, columns=['ngram', 'count'])
    except:
        return pd.DataFrame()
    return count_df


def regex_apply(text_series, regex_dict, language):
    """
    text: texts to be processed (pd.Series)
    regex_dict: dictionary with regex transformers (dict)
    language: chosen language for regex stopwords removal (str)
    return:  pd.Series with treated texts
    """

    #creating pandas dataframe and renaming column to a generic name
    df_aux = pd.DataFrame(text_series)
    col_name = df_aux.columns[0]
    df_aux = df_aux.rename(columns={col_name: 'text'})

    #applying all regex functions in the regex_transformers dictionary
    for regex_name, regex_function in regex_dict.items():
        #print(regex_name)
        df_aux['text'] = df_aux.text.apply(lambda x: regex_function(x) if regex_name != 'stopwords' else regex_function(x, language))
            
    return df_aux['text']


def text_pipeline(text_series, regex_dict, language, vectorizer, stemming_option):
    #Regex process
    df_aux = pd.DataFrame()
    df_aux['treated_text'] = regex_apply(text_series, regex_dict,language)

    #Stemming options 'rslp' or 'Snowball'
    df_aux['stemming'] = df_aux['treated_text'].apply(lambda x: stemming_process(x, 'portuguese','rslp') if stemming_option == 'rslp' else stemming_process(x, language))

    #Feature extraction process
    features, df_features = extract_features_from_corpus(list(df_aux.stemming), vectorizer)

    return df_aux['treated_text'], features, df_features

    
def sentiment_analysis(text, regex_list, language, vectorizer, model):
    # Applying the pipeline
    if type(text) is not list:
        text = [text]
    text_prep = regex_apply(text, regex_transformers,'portuguese')
    matrix = vectorizer.transform(text_prep)
    
    # Predicting sentiment
    pred = model.predict(matrix)
    proba = model.predict_proba(matrix)
    
    #return pred, proba
    #Doing the sentiment and its score
    if pred[0] == 'positive':
        return pred, 100 * round(proba[0][1], 2)
    else:
        return pred, 100 * round(proba[0][0], 2)


    
# plotando wordclouds
def plot_wordcloud(text, title):
    # criando o objeto WordCloud
    nuvem_palavras = WordCloud(width = 1200, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 8).generate(palavras)

    # plotando a wordcloud
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(nuvem_palavras)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.title(f'{title}', fontsize=20)

    plt.show()
    
# plotando multiplos wordlouds lado a lado    
def plot_paired_wordcloud(textos, titulos)
	wordclouds = []
	for i, texto in enumerate(textos):
	    wordcloud = WordCloud(width=1200, height=800, max_words=50,
	    			  background_color="white", stopwords = stopwords,
	    			  min_font_size = 8).generate(texto)
	    wordclouds.append(wordcloud)

	# Plotando as WordClouds lado a lado com seus títulos
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

	for i, ax in enumerate(axes.flatten()):
	    ax.imshow(wordclouds[i], interpolation='bilinear')
	    ax.set_title(titulos[i])
	    ax.axis("off")

	plt.tight_layout()
	plt.show()    
