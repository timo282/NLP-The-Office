import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import contractions

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


PATH = "../data/"
FILE = "the-office-lines_scripts.csv"

# concatenate line_text for each scene
def concatenate_scenes(df):
    if "directionals" in df.columns:
        df["directionals"] = df["directionals"].fillna("")

    df["line_text"] = df["speaker"] + ": " + df["line_text"]
    df = df.drop(columns=["speaker"])
    df = df.groupby(["season", "episode", "scene"]).agg(lambda x: " ".join(x)).reset_index()

    df["season_episode"] = df.apply(lambda x: f"{x['season']}{'0' if x['episode']<10 else ''}{x['episode']}", axis=1)
    return df

def extract_directorals(df):
    # extract text from line_text in square brackets, put it in new column called "directionals", multiple square brackets will be extracted as a list
    df["directionals"] = df["line_text"].str.extractall(r"\[(.*?)\]").unstack().apply(lambda x: ", ".join(x.dropna()), axis=1)
    # delete the extracted text from line_text
    df["line_text"] = df["line_text"].str.replace(r"\[(.*?)\]", "", regex=True).str.strip()
    return df

# bare string preprocessing
def remove_punctuation(df):
    return df["line_text"].str.replace(r"[^\w\s]", "", regex=True)

def lower(df):
    return df["line_text"].apply(lambda x: x.lower())

def remove_stopwords(df):
    stop_words = set(stopwords.words('english'))
    return df["line_text"].apply(lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words]))

def expanding_contractions(df):
    return df["line_text"].apply(lambda x: contractions.fix(x))

def tokenize(df, tokenizer="TreeBankWord", tokenize_specialwords=True, names_csv=PATH+"character_names.csv", compound_words_txt=PATH+"compound_words_the-office_by_chatgpt.txt"):
    if tokenizer=="TreeBankWord":
        t = nltk.tokenize.TreebankWordTokenizer()
    elif tokenizer=="WordPunct":
        t = nltk.tokenize.WordPunctTokenizer()
    elif tokenizer=="Whitespace":
        t = nltk.tokenize.WhitespaceTokenizer()
    else:
        raise ValueError(f"Tokenizer {tokenizer} does not exist.")

    tmp = df["line_text"].apply(lambda x: t.tokenize(x))

    if tokenize_specialwords:
        names = pd.read_csv(names_csv, sep=";", encoding='cp1252').Character.values
        names = names.tolist()
        names.extend([name.lower() for name in names])
        with open(compound_words_txt, "r") as f:
            compound_words = f.read().split(",")
        compound_words = [word.strip() for word in compound_words]
        compound_words.extend([w.lower() for w in compound_words])
        special_words = names + compound_words
        special_tokenizer = MWETokenizer([w.split(" ") for w in special_words])
        return tmp.apply(lambda x: special_tokenizer.tokenize(x))
    
    return tmp

def lemmatize(df):
    wordnet_lemmatizer = WordNetLemmatizer()
    # is working, but not very good results because of the simple speech of the characters
    return df["line_text"].apply(lambda x: " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

def stem(df):
    porter_stemmer = PorterStemmer()
    return df["line_text"].apply(lambda x: " ".join([porter_stemmer.stem(word) for word in word_tokenize(x)]))

# tagging
def pos_tag(df):
    return df["line_text"].apply(lambda x: nltk.pos_tag(word_tokenize(x)))

def preprocess(
        df, 
        concat_scenes=False, 
        extract_direc=False, 
        remove_punct=False, 
        rmv_stopwords=False, 
        lwr=False, 
        exp_contractions=False, 
        conversion:str=None,
        normalize:str=None,
        tokenizer=(None, False, PATH+"character_names.csv", PATH+"compound_words_the-office_by_chatgpt.txt") # parameter for tokenize function (tokenizer(string), tokenize_specialwords(bool)), only used if conversion is "tokenize"
        )->pd.DataFrame:
    
    # remove deleted scenes from script
    df = df[df["deleted"] == False]
    df = df.drop(columns=["deleted"])

    # add "episode id" which is season+episode
    df["season_episode"] = df.apply(lambda x: f"{x['season']}{'0' if x['episode']<10 else ''}{x['episode']}", axis=1)
    
    if (extract_direc):
        df = extract_directorals(df)
    if (concat_scenes):
        df = concatenate_scenes(df)

    if (exp_contractions):
        df['line_text'] = expanding_contractions(df)  
    if (remove_punct):
        df['line_text'] = remove_punctuation(df)
    if (lwr):
        df['line_text'] = lower(df)
    if (rmv_stopwords):
        df['line_text'] = remove_stopwords(df) 

    if (normalize == "lemmatize"):
        df['line_text'] = lemmatize(df)
    elif (normalize == "stem"):
        df['line_text'] = stem(df)   

    if (conversion == "tokenize"):
        df['line_text']  = tokenize(df, tokenizer[0], tokenizer[1], names_csv=tokenizer[2], compound_words_txt=tokenizer[3])
    elif (conversion == "pos_tag"):
        df['line_text'] = pos_tag(df)


    return df

# feature extraction
def extract_features(df, vectorizer):
    if vectorizer == "binary":
        vectorizer = CountVectorizer(binary=True)
    elif vectorizer == "count":
        vectorizer = CountVectorizer() 
    elif vectorizer == "tfidf":
        vectorizer = TfidfVectorizer()
    elif vectorizer == "hashing":
        vectorizer = HashingVectorizer()

    result = vectorizer.fit_transform(df["line_text"])

    return pd.DataFrame(result.todense(), columns=sorted(vectorizer.vocabulary_.keys()))

def feature_selection (feature_df, selection_method):
    # TODO: add feature selection e.g. DF (document frequency)
    print("nothin here yet")