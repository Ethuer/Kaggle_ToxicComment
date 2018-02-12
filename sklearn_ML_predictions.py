
# coding: utf-8

# In[1]:

# python class for Multi ML predictions


# In[ ]:


import pandas as pd, numpy as np
import sys
from time import time
from insult_functions import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier


# In[ ]:

class multi_svm_predictor():
    
    def __init__(self, train_location = '', test_location = '', hashing=True ):
        self.train = pd.read_csv(train_location)
        self.test = pd.read_csv(test_location)
        self.labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        self.hashing = hashing
    
    
    def split_train_val(self):
        msk = np.random.rand(len(corpus)) < 0.95
        self.val = self.train[~msk]
        self.train = self.train[msk]
    
    
    def populate_labels(self):
        self.y_list_train = [self.train[x] for x in self.labels]
        self.y_list_val = [self.val[x] for x in self.labels]
        
    
    def vectorize_data(self):
        if self.hashing:
            self.vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=60000)
            self.X_train = self.vectorizer.transform(self.train['comment_text'])
        else:
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
            self.X_train = self.vectorizer.fit_transform(self.train['comment_text'])
        
        
    
    def fit_vectorization(self, X_data)
        return X_test = self.vectorizer.transform(X_data['comment_text'])
    
    def benchmark(clf):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
    
    #if 'Gaussian' in str(clf) or 'BayesianRidge' in str(clf):
    #    clf.fit(X_train_dense, y_train)
    #else:  ## dense takes massive amounts of memory  better to drop those classifiers
    
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        if 'Lasso' in str(clf) or 'ElasticNet' in str(clf):
            #score = cwlog_singlecolumn(y_test,pred)
            pred = np.round(pred,0)   
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
    
        #print("confusion matrix:")
        #print(metrics.confusion_matrix(y_test, pred))

        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time
    
    
    def predict(clf, X_test, max_count = 10000):
        #clf.fit(X_train, y_train)
    
        combinedout = []
    
        # chunk it
        for part in range( (X_test.shape[0] // max_count) +1 ):
            if X_test.shape[0] >= part*max_count:
                combinedout.append(clf.predict(X_test[ (part*max_count): (part*max_count) + max_count ]))
            #print(pred.shape)
            else:
                combinedout.append(clf.predict(X_test[ (part*max_count):]))
        return np.concatenate(combinedout)
    
    def populate_label( X_train, y_train, X_test, log_prob = False ):
        predictions = []
        train_dat = []
        for clf, name in (
            #(LassoLars(),"LassoLars"),
            #(BayesianRidge(),"BayesianRidge"),
            #(GaussianNB(),"Gaussian NB"), #dense
            (GradientBoostingClassifier(),"Gradient Boosting"),
            (ExtraTreesClassifier(),"ExtraTreesClassifier"),
            (AdaBoostClassifier(),"AdaBoostClassifier"),
            (LinearSVC(),"LinearSVC"),
            (NearestCentroid(),"NearestCentroid"),
            (BernoulliNB(binarize=False, fit_prior=True, alpha=0.1),"BernoulliNB"),
            (Lasso(),"Lasso"),  # regressor
            #(ElasticNet(),"ElasticNet"), # regressor
            #(SGDClassifier(),"SGDClassifier"),
            (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier sag"),
            (Perceptron(max_iter=150), "Perceptron"),
            (PassiveAggressiveClassifier(max_iter=150), "Passive-Aggressive hinge"), # hinge > squarehinge
            (KNeighborsClassifier(n_neighbors=8), "kNN8"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        
        
        
            if log_prob:
                try:
                    predictions.append(predict_logprob(clf, X_train=X_train, X_test=X_test, y_train = y_train))
                except:
                # it's just input data for the Dense layer, so I'll mix log probabs and labels
                    predictions.append(predict(clf, X_train=X_train, X_test=X_test, y_train = y_train))
            else:
            
            
                clf = fit_clf(clf, X_train,y_train)
            
                predictions.append(predict(clf,X_test=X_test))
                train_dat.append(predict(clf, X_test=X_train))
        
        return np.asarray(train_dat), np.asarray(predictions)
    
    def run(self):
        self.train_predict, self.val_predict = populate_label(self.X_train,self.y_train,self.X_test)
        
        train_predict = []
        val_predict = []
        for count, y_train in enumerate(self.y_list_train):
            t_pred,v_pred =  populate_label(self.X_train,self.y_train,self.X_test)
            print("finished label : ", labels[count])
            train_predict.append(t_pred)
            val_predict.append(v_pred)
            
        self.concat_train = np.transpose(np.vstack(train_predict))
        self.concat_val = np.transpose(np.vstack(val_predict))

        self.train_labels = np.transpose(np.vstack(self.y_list_train))
        self.val_labels = np.transpose(np.vstack(self.y_list_val))
    def save(self):
        
        np.save('72_dim_MLarray_train.npy',self.concat_train)
        np.save('72_dim_MLarray_val.npy',self.concat_val)

        np.save('72_dim_MLarray_train_labels.npy',self.train_labels)
        np.save('72_dim_MLarray_val_labels.npy',self.val_labels)

    def restore(self):
        self.concat_train = np.load('72_dim_MLarray_train.npy')
        self.concat_val = np.load('72_dim_MLarray_val.npy')
        self.train_labels = np.load('72_dim_MLarray_train_labels.npy')
        self.val_labels = np.load('72_dim_MLarray_val_labels.npy')
        
    def run_predict(self, X_test):
        """ expects a pandas dataframe ,  vectorizes it and runs available clfs on it, returns predictions """
        # ToDo  write the function
        

