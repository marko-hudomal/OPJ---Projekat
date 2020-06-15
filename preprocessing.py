import pandas as pd
import re
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

COMMENT = "Comment"


def read_file(file_name):
    return pd.read_csv(file_name, encoding='ISO-8859-1', delimiter="\t")


def process_data(file_df, to_lower_case=False, remove_stop_words=False, stem=False, spec_chars_to_keep="@_$=;-", comment_index_to_analize=None):
    df = file_df.copy(deep=True)
    for index, comment in enumerate(df[COMMENT]):
        if comment_index_to_analize == index:
            print("READ FROM FILE\n", comment, "\n")

        comment = re.sub("\\\\n", " ", comment)  # remove \n
        if comment_index_to_analize == index:
            print("REMOVED \\n\n", comment, "\n")

        words = word_tokenize(comment)
        if comment_index_to_analize == index:
            print("TOKENIZATION\n", words, "\n")

        if to_lower_case:
            words = [word.lower() for word in words]
            if comment_index_to_analize == index:
                print("LOWER CASE\n", words, "\n")

        regex = "[^A-Za-z{}]".format(spec_chars_to_keep)
        words = [re.sub(regex, "", word) for word in words]  # remove characters
        words = [word for word in words if word]  # remove empty string
        if comment_index_to_analize == index:
            print("REGEX\n", words, "\n")

        if remove_stop_words:
            words = [word for word in words if word not in stopwords.words("english")]
            if comment_index_to_analize == index:
                print("STOP WORDS\n", words, "\n")

        if stem:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
            if comment_index_to_analize == index:
                print("STEM\n", words, "\n")

        if comment_index_to_analize == index:
            print("RESULT\n", " ".join(words), "\n")

        df.loc[index, COMMENT] = " ".join(words)
    return df


def vectorize(df, max_features=1000, ngram_range=(1, 1), tf=False, tfidf=False):
    pipe = Pipeline([('count', CountVectorizer(max_features=max_features, ngram_range=ngram_range, tokenizer=(lambda s: s.split()))),
                     ('tfidf', TfidfTransformer(use_idf=not tf))]).fit(df[COMMENT])

    feature_names = pipe['count'].get_feature_names()
    if tf:
        return pipe.transform(df[COMMENT]).toarray(), feature_names
    if tfidf:
        return pipe.transform(df[COMMENT]).toarray(), feature_names

    return pipe['count'].transform(df[COMMENT]).toarray(), feature_names


if __name__ == "__main__":

    file_df = read_file("input.txt")
    df = process_data(file_df, to_lower_case=True, remove_stop_words=False,
                      stem=False, spec_chars_to_keep="@_$=;-", comment_index_to_analize=1)

    counts, names = vectorize(df)
    tfidfs, names = vectorize(df, max_features=500, tfidf=True)
    tfs, names = vectorize(df, max_features=300, tf=True)
