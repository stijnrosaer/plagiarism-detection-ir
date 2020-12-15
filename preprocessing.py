import pandas as pd
import numpy as np
import random
from collections import defaultdict
from IPython.display import clear_output
import inflect


def preprocess(df, filename):
    import re
    df["article"] = df["article"].str.lower()  # to lower case

    df['article'] = df['article'].map(
        lambda x: re.sub(r'[,!.;+-@!%^&*)(_\\\'\"“”’—]', '', str(x)))  # remove special characters
    df['article'] = df['article'].map(lambda x: re.sub(r'\W+', ' ', str(x)))  # remove exess white spaces
    df['article'] = df['article'].map(
        lambda x: re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', str(x)))  # remove single letters

    # p = inflect.engine()
    #
    # def to_singular(word):
    #     sing = p.singular_noun(word)
    #     if not sing:
    #         return word
    #     return sing
    #
    # df['article'] = df['article'].map(lambda x: ' '.join([to_singular(word) for word in x.split()]))
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    nltk.download('wordnet')
    stemmer = PorterStemmer()
    df['article'] = df['article'].map(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    lemma = WordNetLemmatizer()
    df['article'] = df['article'].map(lambda x: ' '.join([lemma.lemmatize(word) for word in x.split()]))



    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words("english")
    df['article'] = df['article'].map(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    df.to_csv("preprocess-" + filename, index=False)


if __name__ == "__main__":
    filename = "news_articles_small.csv"
    df = pd.read_csv(filename)
    preprocess(df, filename)