
# coding: utf-8

# In[1]:

# helperfunctions for the RNN_DEEP network


# In[ ]:
import numpy as np


# find repetitions:
def uppercase_letters_ratio(wordlist, testratio):
    
    # lets add 2,5,10,15 and 20 as ohot features
    ucl = 0
    sentencelength=0
    
    output = []
    
    for string in wordlist:
        if string and type(string) != 'NoneType':
            if not 'REDIRECT' in string.upper():
                ucl += sum(1 for c in string if c.isupper())
                sentencelength += len(string)

    if sentencelength > 0:
        if (float(ucl)/ float(sentencelength)) > testratio:
            return 1
    return 0
    #return True  # redirect is a machine command. 

def repetition(wordlist):
    worddict = {}
    
    for element in wordlist:
        if not element in worddict:
            worddict[element] = 0
        worddict[element] +=1
    
    for key,val in worddict.items():
        if val > 7:
            return 1
        if len(key)> 40: # repetition words without spaces
            return 1
    
    return 0


def extend_dataframe_ohotmeasures(element, common_list_bad,rare_list_bad, capsratios = [0.02,0.05,0.07,0.1,0.15,0.2,0.4]):
    """ add a new column to the dataframe, containing the various engineered features
        such as :  percent of capitalization, repetitive sequences and various lists of cursewords"""
    
    ## remove all non alphanumeric characters, but keep spaces,  no need for dots or commas
    element = element = ''.join(letter for letter in element if letter == ' ' or letter.isalnum())
    
    # keep track, so use a list
    outlist = []
    
    # does the comment contain repetitions ? Muhahahaha or just copy paste words. 
    for capsratio in capsratios:
        outlist.append(uppercase_letters_ratio(element.split(' '), capsratio))
    
    element = element.lower()  # now I don't need the caps information anymore 
    
    
    # does the comment contain extensive repetitions ?
    outlist.append(repetition(element.split(' ')))

    
    # test the two lists,  common curses get their own one hot marker,  rare ones get one for all
    
    for word in common_list_bad:
        if word in element:
            outlist.append(1)
        else:
            outlist.append(0)
            
            
    found = 0
    for word in rare_list_bad:
        if word in element:
            found = 1
    outlist.append(found)
    
    return outlist
    
    
    


# In[ ]:

def not_number(word):
    try:
        float(word)
        return False
    except ValueError:
        return True

def clean_words(word):
    """ remove special characters and numbers from words"""
    #  "== is a tag ,  useful information since they are headers
    if '== ' in word:
        # use the word tag as tag,  why not ?
        return 'tag'
        
    outword = ''.join(letter for letter in word if letter.isalnum())
    
    if outword == 'i' or outword == 'a' or len(outword) > 1 :
        if not_number(word):
            return outword
    

def clean_sentence(sentence):
    """ turn a sentence into a list of cleaned words """
    outsentence = []
    
    #print(len(sentence.split(' ')))
    for word in sentence.split(' '):
        outsentence.append(clean_words(word))
        
    #if len(outsentence) > 1:
    #print(len(sentence.split(' ')), len(outsentence))
    return outsentence
    
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def make_dict_of_tokenization(statement,dictofwords):
    """ generate dict of words,  uses nltk tokenization """

    for token in nltk.word_tokenize(statement):
        if not token in dictofwords:
            dictofwords[token] = [len(dictofwords),0] # index lenght of words
        if token in dictofwords:
            dictofwords[token][1] +=1
                
    return nltk.word_tokenize(statement)


def make_dict_of_words(statement,dictofwords):
    """ generate dict of words,  returns cleaned sentences """
    outsentence = []
    for sentence in statement.split('.'):  # toknize better than this
        cleansentence = clean_sentence(sentence)
        for word in cleansentence:
            
            if not word in dictofwords:
                dictofwords[word] = [len(dictofwords),0] # index lenght of words
            if word in dictofwords:
                dictofwords[word][1] +=1

            outsentence.append(dictofwords[word][0]) # generate the integer list for RNN

    return outsentence
            
def purgerarewords(dictionary, minvalue):
    outdict = dict()
    
    for key,value in dictionary.items():
        if value[1] > minvalue:
            outdict[key] = value[0]
    return outdict
    
def useonlycommonwords(statement,dictofwords):
    """ use a dictionary that contains only common words """
    outsentence = []
    for sentence in statement.split('.'):
        cleansentence = clean_sentence(sentence)
        for word in cleansentence:
            if word in dictofwords:
                outsentence.append(word)
            else:
                outsentence.append(0)
    return outsentence


# In[ ]:

import csv
import sys


def load_embedding_dictionary(path_to_vec, limited_dict):
    converse_dict = dict()
#'../embeddings/alt/crawl-300d-2M.vec'
    with open(path_to_vec ) as crawl:
    	infile = csv.reader(crawl, delimiter = ' ',quoting=csv.QUOTE_NONE)
    #infile.next()
    
    	for row in infile:
        #print(row[0])
        	if row[0] in limited_dict:
        	    converse_dict[row[0]] = row[1:-1] # last column contains empty
        	if row[0] == 'None':
        	    converse_dict[row[0]] = row[1:-1]
        	if row[0] == '.':
            	    converse_dict['0'] = row[1:-1]
            
    return converse_dict
    


# In[ ]:

def make_zero_vector(embedding_dims):
    output = []
    for dim in range(embedding_dims):
        output.append(0)
    return output

zerovector = make_zero_vector(300)


# In[ ]:

def get_y_batch(row_to_use):
    outlist = []
    
    #toxic 	severe_toxic 	obscene 	threat 	insult 	identity_hate 	
    outlist.append(row_to_use['toxic'])
    outlist.append(row_to_use['severe_toxic'])
    outlist.append(row_to_use['obscene'])
    outlist.append(row_to_use['threat'])
    outlist.append(row_to_use['insult'])
    outlist.append(row_to_use['identity_hate'])
    
    
    return outlist


def stringlist_to_embed(stringlist,embed_dict, zerovector,desired_len):
    stringlist_len = len(stringlist)
    if stringlist_len > desired_len:
        stringlist_len = desired_len
        
    baselist = [embed_dict[ str(stringlist[s])] for s in range(stringlist_len) ] 
    
    
    while len(baselist) < desired_len:
        baselist.append(zerovector)
    return baselist

def getbatch(corpus,batchsize,embed_dict , zerovector , max_len = 150):
    
    batch = corpus.sample(batchsize)
    y_batch = batch[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    
    OHOT_batch = batch['ohot_engineered'].apply(lambda x: [ohot for ohot in x]).tolist()
    RNN_batch = batch['cleansentences'].apply(lambda x: np.asarray(stringlist_to_embed(x,embed_dict,zerovector,max_len))).tolist()
    seq_len = batch['cleansentences'].apply(lambda x: len(x)).tolist()
    
    return np.asarray(OHOT_batch, dtype=np.int32) ,np.asarray(RNN_batch), seq_len, np.asarray(y_batch, dtype= np.int32)
    
def getbatch_RNN(corpus,batchsize, embed_dict , zerovector , max_len = 150):
    
    batch = corpus.sample(batchsize)
    RNN_batch = batch['cleansentences'].apply(lambda x: np.asarray(stringlist_to_embed(x,embed_dict,zerovector,max_len))).tolist()
    seq_len = batch['cleansentences'].apply(lambda x: len(x)).tolist()
    y_batch = batch[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    
    return RNN_batch, seq_len, np.asarray(y_batch, dtype= np.int32)

def getbatch_OHOT(corpus,batchsize):
    
    batch = corpus.sample(batchsize)
    y_batch = batch[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    
    X_batch = batch['ohot_engineered'].apply(lambda x: [ohot for ohot in x]).tolist()
    return np.asarray(X_batch, dtype=np.int32),  np.asarray(y_batch, dtype= np.int32)


# In[ ]:


def stringlist_to_embed_deep(stringlist,embed_dict_deep, zerovector,desired_len):
    stringlist_len = len(stringlist)
    if stringlist_len > desired_len:
        stringlist_len = desired_len
        
    baselist = [embed_dict_deep[ str(stringlist[s])[0]][str(stringlist[s])] for s in range(stringlist_len) ] 
    
    while len(baselist) < desired_len:
        baselist.append(zerovector)
    return baselist


# In[2]:

# evaluationfunctions


# In[ ]:

from sklearn.metrics import log_loss
import numbers

def column_wise_log_loss(y,y_pred, col_list):
    result = []
    for element in range(len(y)):
        #print("here ", y[element], "  :  ", y_pred[element])
        #res = y[element] - y_pred[element]
        res =  np.sqrt((y[element] - y_pred[element])*(y[element] - y_pred[element]))
        if res!= 0 :
            try:
                result.append(np.log( res ))
            except:
                print(res)
        else:
            result.append(0)
        
    return np.mean(result)

def combine_batch(resultlist,y):
    #for element in y:
    for label in range(len(y)):
        resultlist[label].append( np.float32(y[label]))
    return resultlist
            
def combine_batch_zeroes(resultlist,y):
    #for element in y:
    for label in range(len(y)):
        resultlist[label].append(0.5)
    return resultlist  
    
    
    
def cwlog_batch(y,y_pred, nomean = False):
    # toxic,severe_toxic,obscene,threat,insult,identity_hate
    predlist = [[],[],[],[],[],[]]
    truelist = [[],[],[],[],[],[]]
    result = []
    for element in range(len(y)):
        predlist = combine_batch(predlist,y_pred[element])
        truelist = combine_batch(truelist,y[element])

    # get the average log loss per column
    for label in range(len(truelist)):
        # use the sklearn implementation
        #try:
        #print(y[label], " : ", y_pred[label])
        try:
            logloss = log_loss(y_pred=predlist[label], y_true=truelist[label])
            result.append(logloss)
        except:
            pass
            #print('interrupted extension')
        
        
        #except:
        #    
    
    #print(predlist[0] , truelist[0])
    if not nomean:
        return np.mean(result)
    if nomean:
        return [ str(round(x,3)) for x in result]


def floattobinary(predictions):
    newpreds = []
    for element in predictions:
        elmnt = []
        for label in element:
            if label < 0:
                elmnt.append(0)
            if label > 1:
                elmnt.append(1)
            if label >=0 and label <=1:
                elmnt.append(label)
                
        newpreds.append(elmnt)
    return newpreds


# In[3]:

# various legacy 


# In[ ]:

# prefetch data for the RNN 
def make_zero_vector(embedding_dims):
    output = []
    for dim in range(embedding_dims):
        output.append(0)
    return output


def sent_to_embedding(sentence_aslist,zerovector, embed_dict, max_len):
    outlist = []
    for word in range(max_len):
        
        if word < len(sentence_aslist) and sentence_aslist[word] in embed_dict:
            outlist.append(embed_dict[sentence_aslist[word]])
        else:
            outlist.append(zerovector)
    return np.asarray(outlist, dtype=np.float32)

def get_y_batch(row_to_use):
    outlist = []
    
    #toxic 	severe_toxic 	obscene 	threat 	insult 	identity_hate 	
    outlist.append(row_to_use['toxic'].tolist()[0])
    outlist.append(row_to_use['severe_toxic'].tolist()[0])
    outlist.append(row_to_use['obscene'].tolist()[0])
    outlist.append(row_to_use['threat'].tolist()[0])
    outlist.append(row_to_use['insult'].tolist()[0])
    outlist.append(row_to_use['identity_hate'].tolist()[0])
    
    
    return outlist

def get_y_batch_alt(row_to_use):
    outlist = []
    
    #toxic 	severe_toxic 	obscene 	threat 	insult 	identity_hate 	
    outlist.append(row_to_use['toxic'])
    outlist.append(row_to_use['severe_toxic'])
    outlist.append(row_to_use['obscene'])
    outlist.append(row_to_use['threat'])
    outlist.append(row_to_use['insult'])
    outlist.append(row_to_use['identity_hate'])
    
    
    return outlist





def pandastonumpy(corpus,  embed_dict , zerovector , max_len = 150):
    
    
    X_dat = []
    y_dat = []
    seq_length = []
    
    for rowtouse in corpus.iterrows():
        rowtouse = rowtouse[1]
        y_dat.append(get_y_batch_alt(rowtouse))
        sentence = rowtouse['cleansentences']
        sentence_as_embed = sent_to_embedding(sentence ,zerovector, embed_dict=embed_dict,max_len = max_len )
        #print(type(sentence_as_embed))
        X_dat.append( sentence_as_embed )
        seq_length.append(len(rowtouse['cleansentences']))
        #seq_length_batch.append(max_len)
                                
    return X_dat,y_dat, np.asarray(seq_length, dtype=np.int32)

import random

def split_train_val(X_all,y_all,seq_all,ratio = 0.95):
    X_train = []
    y_train = []
    seq_train = []
    
    X_val = []
    y_val = []
    seq_val = []
    for element in range(len(X_all)):
        if random.uniform(0.0, 1.0) <= ratio:
            X_train.append(X_all[element])
            y_train.append(y_all[element])
            seq_train.append(seq_all[element])
        else:
            X_val.append(X_all[element])
            y_val.append(y_all[element])
            seq_val.append(seq_all[element])
    
    return X_train,y_train,seq_train,X_val,y_val,seq_val
    
def get_batch_np(X_all,y_all,seq_all,batchsize = 200):
    X_batch = []
    y_batch = []
    seq_batch = []
    
    for sample in range(batchsize):
        try:
            element = random.randint(0,(len(X_all)-1))
            X_batch.append(X_all[element])
            y_batch.append(y_all[element])
            seq_batch.append(seq_all[element])
        except:
            print('failed to load a sample')
    return X_batch,y_batch,seq_batch

def logloss(true_label, predicted, eps):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -log(p)
    else:
        return -log(1 - p)



