import re
import numpy as np
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


nlp = spacy.load("en_core_web_sm")
scaler = MinMaxScaler()



class Autoscraper():
    def __init__(self):
        self.main_matrix = np.empty([0,0], dtype=float)

    def binarize(self, x):
        if x:
            return 1
        else:
            return 0

    def parse_examples(self, examples):

        word_counts = []
        character_counts = []
        information_for_regression_list = []
        alphas = []
        stopws = []
        n_digits = []

        for example in examples:
            word_count = len(example.split())
            character_count = len(example)
            information_for_regression = [[]] * 5
            alpha = []
            stopw = []
            pipeline_information =  nlp(example)

            for token in pipeline_information:
                information_for_regression[0][0].append(token.text)
                information_for_regression[0][1].append(token.dep_)
                information_for_regression[0][2].append(token.pos_)
                information_for_regression[0][3].append(token.tag_)
                information_for_regression[0][4].append(token.shape_)
                alpha.append(token.is_alpha)
                stopw.append(token.is_stop)

            alpha = sum(list(map(self.binarize, alpha)))
            stopw = sum(list(map(self.binarize, stopw)))

            word_counts.append([word_count])
            character_counts.append([character_count])
            information_for_regression_list.append(information_for_regression)
            alphas.append([alpha])
            stopws.append([stopw])
            n_digits.append([len(re.findall(r"[0-9]", example))])

            return word_counts, character_counts, information_for_regression_list, alphas, stopws, n_digits


    def fit(self, examples=None, labels=None, regression='Ridge'):

        word_counts, character_counts, information_for_regression_list, alphas, stopws, n_digits = self.parse_examples(examples)

        self.text_vectorizer = CountVectorizer(max_features=4000, ngram_range = (1, 1))
        X = [info[0] for info in information_for_regression_list]
        X = self.text_vectorizer.fit_transform(X)
        Y = labels
        self.text_model = LogisticRegression()
        self.text_model.fit(X, Y)
        c = self.text_model.predict_proba(X)
        text_scores = np.array([d[1] for d in c]).reshape(-1, 1)


        self.dep_vectorizer = CountVectorizer(ngram_range = (1, 1))
        X = [info[1] for info in information_for_regression_list]
        X = self.dep_vectorizer.fit_transform(X)
        self.dep_model = LogisticRegression()
        self.dep_model.fit(X, Y)
        c = self.dep_model.predict_proba(X)
        dep_scores = np.array([d[1] for d in c]).reshape(-1, 1)


        self.pos_vectorizer = CountVectorizer(ngram_range = (1, 1))
        X = [info[2] for info in information_for_regression_list]
        X = self.pos_vectorizer.fit_transform(X)
        self.pos_model = LogisticRegression()
        self.pos_model.fit(X, Y)
        c = self.pos_model.predict_proba(X)
        pos_scores = np.array([d[1] for d in c]).reshape(-1, 1)


        self.tag_vectorizer = CountVectorizer(ngram_range = (1, 1))
        X = [info[3] for info in information_for_regression_list]
        X = self.tag_vectorizer.fit_transform(X)
        self.tag_model = LogisticRegression()
        self.tag_model.fit(X, Y)
        c = self.tag_model.predict_proba(X)
        tag_scores = np.array([d[1] for d in c]).reshape(-1, 1)


        self.shape_vectorizer = CountVectorizer(max_features=1000, ngram_range = (1, 1))
        X = [info[4] for info in information_for_regression_list]
        X = self.shape_vectorizer.fit_transform(X)
        self.shape_model = LogisticRegression()
        self.shape_model.fit(X, Y)
        c = self.shape_model.predict_proba(X)
        shape_scores = np.array([d[1] for d in c]).reshape(-1, 1)


        word_counts = scaler.fit_transform(word_counts)
        character_counts = scaler.fit_transform(character_counts)
        alphas = scaler.fit_transform(alphas)
        stopws = scaler.fit_transform(stopws)
        n_digits = scaler.fit_transform(n_digits)


        Xt = np.concatenate([text_scores, dep_scores, pos_scores, tag_scores, shape_scores, word_counts, character_counts, alphas, stopws, n_digits],
                            axis=1)
        
        if type(regression) == str:
            if regression == 'Ridge':
                self.Tmodel = Ridge(alpha=1.0)
                self.Tmodel.fit(Xt, Y)
            if regression == 'logistic':
                self.Tmodel = LogisticRegression()
                self.Tmodel.fit(Xt, Y)
        else:
            self.Tmodel = regression()
            self.Tmodel.fit(Xt, Y)


    def predict(self, examples):

        word_counts, character_counts, information_for_regression_list, alphas, stopws, n_digits = self.parse_examples(examples)

        X = [info[0] for info in information_for_regression_list]
        X = self.text_vectorizer.transform(X)
        c = self.text_model.predict_proba(X)
        text_scores = np.array([d[1] for d in c]).reshape(-1, 1)

        
        X = [info[1] for info in information_for_regression_list]
        X = self.dep_vectorizer.transform(X)
        c = self.dep_model.predict_proba(X)
        dep_scores = np.array([d[1] for d in c]).reshape(-1, 1)


        X = [info[2] for info in information_for_regression_list]
        X = self.pos_vectorizer.transform(X)
        c = self.pos_model.predict_proba(X)
        pos_scores = np.array([d[1] for d in c]).reshape(-1, 1)

        
        X = [info[3] for info in information_for_regression_list]
        X = self.tag_vectorizer.transform(X)
        c = self.tag_model.predict_proba(X)
        tag_scores = np.array([d[1] for d in c]).reshape(-1, 1)

        
        X = [info[4] for info in information_for_regression_list]
        X = self.shape_vectorizer.transform(X)
        c = self.shape_model.predict_proba(X)
        shape_scores = np.array([d[1] for d in c]).reshape(-1, 1)


        word_counts = scaler.fit_transform(word_counts)
        character_counts = scaler.fit_transform(character_counts)
        alphas = scaler.fit_transform(alphas)
        stopws = scaler.fit_transform(stopws)
        n_digits = scaler.fit_transform(n_digits)


        Xt = np.concatenate([text_scores, dep_scores, pos_scores, tag_scores, shape_scores, word_counts, character_counts, alphas, stopws, n_digits],
                            axis=1)
        
        return self.Tmodel.predict(Xt)

    