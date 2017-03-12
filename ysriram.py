# import packages
import json
import os
import numpy as np
import pandas as pd
import random
import nltk 
from nltk import PorterStemmer
import re
import string
import math
import operator    
import time

#change the directiory
# os.chdir('')

#We preprocess the data and create the necessary dictionaries:
random.seed(99)
dataframe('yelp_academic_dataset_review.json')

start_pre = time.time()
preprocessing(df)
end_pre = time.time()
pretime = end_pre - start_pre

start_dict = time.time()
dictionaries(df)
end_dict = time.time()
dicttime = end_dict - start_dict

del df

#Using the dictionaries, use the Knn and Rocchio functions with various similarity calculations and similarity measures:
##ROCCHIO
start_rocidf = time.time() ##rocchio's with tfidf weights, cosine similarity
roc(w = tf_idf, sim = 'cosine')
end_rocidf = time.time()
rocidftime = end_rocidf - start_rocidf

start_roctf = time.time() ##rocchio's with tf weights, cosine similarity
roc(w = tf, sim = 'cosine')
end_roctf = time.time()
roctftime = end_roctf - start_roctf

start_rocbinary = time.time() ##rocchio's with binary weights, cosine similarity
roc(w = binary, sim = 'cosine')
end_rocbinary = time.time()
rocbinarytime = end_rocbinary - start_rocbinary

start_drocidf = time.time() ##rocchio's with tfidf weights, dice similarity
roc(w = tf_idf, sim = 'dice')
end_drocidf = time.time()
rocdidftime = end_drocidf - start_drocidf

start_droctf = time.time() ##rocchio's with tf weights, dice similarity
roc(w = tf, sim = 'dice')
end_droctf = time.time()
rocdtftime = end_droctf - start_droctf

start_drocbinary = time.time() ##rocchio's with binary weights, dice similarity
roc(w = binary, sim = 'dice')
end_drocbinary = time.time()
rocdbinarytime = end_drocbinary - start_drocbinary



#5NN
start_5nnidf = time.time() ##5-NN with tf_idf weights, cosine similarity
knn(k=5, w=tf_idf, sim = 'cosine')
end_5nnidf = time.time()
idf5nntime = end_5nnidf - start_5nnidf

start_5nntf = time.time() ##5-NN with tf weights, cosine similarity
knn(k=5, w=tf, sim = 'cosine')
end_5nntf = time.time()
tf5nntime = end_5nntf - start_5nntf

start_5nnbinary = time.time() ##5-NN with binary weights, cosine similarity
knn(k=5, w=binary, sim = 'cosine')
end_5nnbinary = time.time()
binary5nntime = end_5nnbinary - start_5nnbinary

start_d5nnidf = time.time() ##5-NN with tf_idf weights, dice similarity
knn(k=5, w=tf_idf, sim = 'dice')
end_d5nnidf = time.time()
idfd5nntime = end_d5nnidf - start_d5nnidf

start_d5nntf = time.time() ##5-NN with tf weights, dice similarity
knn(k=5, w=tf, sim = 'dice')
end_d5nntf = time.time()
tfd5nntime = end_d5nntf - start_d5nntf

start_d5nnbinary = time.time() ##5-NN with binary weights, dice similarity
knn(k=5, w=binary, sim = 'dice')
end_d5nnbinary = time.time()
binaryd5nntime = end_d5nnbinary - start_d5nnbinary

#9NN
start_9nnidf = time.time() ##9-NN with tf_idf weights, cosine similarity
knn(k=9, w=tf_idf, sim = 'cosine')
end_9nnidf = time.time()
idf9nntime = end_9nnidf - start_9nnidf

start_9nntf = time.time() ##9-NN with tf weights, cosine similarity
knn(k=9, w=tf, sim = 'cosine')
end_9nntf = time.time()
tf9nntime = end_9nntf - start_9nntf

start_9nnbinary = time.time() ##9-NN with binary weights, cosine similarity
knn(k=9, w=binary, sim = 'cosine')
end_9nnbinary = time.time()
binary9nntime = end_9nnbinary - start_9nnbinary

start_d9nnidf = time.time() ##9-NN with tf_idf weights, dice similarity
knn(k=9, w=tf_idf, sim = 'dice')
end_d9nnidf = time.time()
idfd9nntime = end_d9nnidf - start_d9nnidf

start_d9nntf = time.time() ##9-NN with tf weights, dice similarity
knn(k=9, w=tf, sim = 'dice')
end_d9nntf = time.time()
tfd9nntime = end_d9nntf - start_d9nntf

start_d9nnbinary = time.time() ##9-NN with binary weights, dice similarity
knn(k=9, w=binary, sim = 'dice')
end_d9nnbinary = time.time()
binaryd9nntime = end_d9nnbinary - start_d9nnbinary

#31NN
start_31nnidf = time.time() ##31-NN with tf_idf weights, cosine similarity
knn(k=31, w=tf_idf, sim = 'cosine')
end_31nnidf = time.time()
idf31nntime = end_31nnidf - start_31nnidf

start_31nntf = time.time() ##31-NN with tf weights, cosine similarity
knn(k=31, w=tf, sim = 'cosine')
end_31nntf = time.time()
tf9nntime = end_31nntf - start_31nntf

start_31nnbinary = time.time() ##31-NN with binary weights, cosine similarity
knn(k=9, w=binary, sim = 'cosine')
end_31nnbinary = time.time()
binary31nntime = end_31nnbinary - start_31nnbinary

start_d31nnidf = time.time() ##31-NN with tf_idf weights, dice similarity
knn(k=9, w=tf_idf, sim = 'dice')
end_d31nnidf = time.time()
idfd9nntime = end_d31nnidf - start_d31nnidf

start_d31nntf = time.time() ##31-NN with tf weights, dice similarity
knn(k=9, w=tf, sim = 'dice')
end_d31nntf = time.time()
tfd31nntime = end_d31nntf - start_d31nntf

start_d31nnbinary = time.time() ##31-NN with binary weights, dice similarity
knn(k=9, w=binary, sim = 'dice')
end_d31nnbinary = time.time()
binaryd31nntime = end_d31nnbinary - start_d31nnbinary


#******************************************************************************
#provide a path to a json file...will open and return a dataframe**************
#******************************************************************************
def dataframe(path):
    '''This function takes the path of the json file as the input, parses it, and creates a
    pandas dataframe with the the following columns: review ID, review, and rating
    '''
    
    file = open(path, 'r')
    dataAll = file.read().split('\n')
    
    #get a 10,000 row sample
    data = random.sample(dataAll, 10000)
    
    #create the idd, review and ratings empty lists
    idd = []
    reviews = []
    ratings = []
    
    #extract the entries within the data sample of 10k
    for entry in data:
        extract = json.loads(entry)
        idd.append(extract['review_id'])
        reviews.append(extract['text'])
        ratings.append(extract['stars'])
        
    
    #create a dataframe of the json data
    data_dict = {'id':idd,'reviews': reviews, 'ratings': ratings}
    global df
    df = pd.DataFrame(data_dict);

 
#******************************************************************************
#returns the original dataframe, with stemmed text*****************************
#******************************************************************************
def preprocessing(df):
    '''This function takes the pandas dataframe and preprocesses the text column
    by removing stop words and punctuations, stems each word using 
    Porter stemming algorithm, and removes all those words that are less than 4 letters long.
    '''
       
    stopWords = ['a','able','about','across','after','all','almost','also',
                 'am','among','an','and','any','are','as','at','be','because',
                 'been','but','by','can','cannot','could','dear','did','do',
                 'does','either','else','ever','every','for','from','get','got'
                 ,'had','has','have','he','her','hers','him','his','how',
                 'however','i','if','in','into','is','it','its','just','least'
                 ,'let','like','likely','may','me','might','most','must','my',
                 'neither','no','nor','not','of','off','often','on','only',
                 'or','other','our','own','rather','said','say','says','she'
                 ,'should','since','so','some','than','that','the','their',
                 'them','then','there','these','they','this','tis','to',
                 'too','twas','us','wants','was','we','were','what','when',
                 'where','which','while','who','whom','why','will','with',
                 'would','yet','you','your']
             
    punctuations = ['"'," ",".","/",";","'","?","&","-", ",","!", "]", "["]

    collect = []
    
    for i in df.index:
        text = df.xs(i)['reviews'].lower().strip()
        textLst = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*",text)
        
        countof = True    
        
        while countof:
            y = 0
            for i,j in enumerate(textLst): 
                
                textLst[i] = PorterStemmer().stem_word(j.lower())
                
                if j in stopWords or j in punctuations or len(j) <= 3 or j.isdigit():
                    textLst.pop(i)
                    y = y+1
                    continue
                for c in punctuations:
                    if c in j:
                        textLst.pop(i)
                        y = y+1
                        break
            if y == 0:
                countof = False
        
        collect.append(" ".join(textLst))
    
    df['reviews'] = collect;

#******************************************************************************
#create an inverted index******************************************************
#******************************************************************************
def dictionaries(df):
    '''This function takes the preprocessed dataframe and creates multiple inverted indices
    as dictionaries. A rating dictionary with the ID and rating values; a tf dictionary
    with words, postings, and their corresponding term frequencies; a tf_idf dictionary
    with words, postings, and their corresponding tfxidf values;  a binary dictionary
    with words, postings, and their corresponding binary weights; inverted_index dictionary with
    the words and their corresponding postings; norm_tf with documents with their normalization 
    factors using term frequencies; a norm_binary dictionary with documents and their normalization 
    factors using binary weight values; and  norm_tf_idf with documents and their normalization 
    factors using tfxidf values. This function also creates a list of documents that are used as the
    training documents (805 of the total documents) and a list that are used as the testing documents
    (20% of the total documents).
    '''
    
    #initalize the needed dictionaries
    global rating
    global tf
    global tf_idf
    global inverted_index
    global trainDoc
    global testDoc
    global norm_tf
    global norm_tf_idf
    global norm_binary
    global binary
    
    rating = {}
    tf_idf = {}
    norm = {}
    inverted_index = {}
    tf = {}
    norm_tf_idf = {}
    norm_2 = {}
    norm_tf = {}
    norm_3 = {}
    norm_binary = {}
    binary = {}
    
    for index,row in df.iterrows():
        
        #put each column in a list
        review_id = row[0]
        review_text = row[2]
        review_text = review_text.split(' ')
        review_rating = row[1]
    
        #add to the reviews dictionary
        rating[review_id] = review_rating
        norm_2[review_id] = []
        norm[review_id] = []
        norm_3[review_id] = []
        
        #training and testing split, 80%-20%
        docLst = np.array(list(rating.keys()))
        
        #np.random.shuffle(docLst)
        trainDoc = docLst[:(round(0.8*len(docLst)))]
        testDoc = docLst[(round(0.8*len(docLst))):]
    
        #create the inverted index {term : [doct list]}
        for word in review_text:
            
            if word not in inverted_index.keys():
                inverted_index[word] = [review_id]
            else:
                inverted_index[word].append(review_id)
                
        for key,value in inverted_index.items():
            inverted_index[key] =  list(set(value))
        

    for index,row in df.iterrows():
        
        #put each column in a list
        review_id = row[0]
        review_text = row[2]
        review_text = review_text.split(' ')
        review_rating = row[1]    
        
        for word in review_text:
            
            binary[(word,review_id)] = 1
            
            #tf dictionary
            counter = 0
            counter = review_text.count(word)
            tf[(word,review_id)] = counter
                            
            #get idf values
            count = len(rating)
            length = len(inverted_index[word])
            num = count/length
            idf = math.log(num,2)
                
            #tf_idf postings
            posting = idf * counter
            tf_idf[(word, review_id)] = posting   
            
            
    for key,values in tf.items():
        norm_2[key[1]].append(values)
        
    for key,values in tf_idf.items():
        norm[key[1]].append(values)
        norm_3[key[1]].append(1)
        


    for key,value in norm_2.items():
        norm_tf[key] = sum(i*i for i in value)
            
    for key, value in norm.items():
        norm_tf_idf[key] = sum(i*i for i in value)
        
    for key,value in norm_3.items():
        norm_binary[key] = sum(i*i for i in value)
    
    

####VERSION 1: CATEGORIZING DOCUMENTS USING ROCCHIOS WITH TFXIDF TERM WEIGHTS AND COSINE SIMILARITY
                
#******************************************************************************
#Rochios Method - First getcentroids vectors for each class********************
#******************************************************************************

def roc(w = tf_idf, sim = 'cosine'):
    '''This function calculates the document categories for the reviews using Rocchio's method.
    It takes in the term weights (tf, tf_idf, or binary) and similarity method (cosine or dice) as the parameters.
    It calculates prototype vectors each of the unique ratings and then calculates the similarity between these
    vectors and the documents in the testing dataset to predict the rating category into which the testing documents 
    would fall. After this, it calculates the accuracy by comparing the predicted category to the actual document 
    category.
    '''
    
    global protoVec
    global termw
    global rocSim
    global accuracyRoc   
    global predActRating
    
    if w == tf_idf:
        n = norm_tf_idf
        
    if w == tf:
        n = norm_tf
        
    if w == binary:
        n = norm_binary
        
    
    docs_rating = {}
    protoVec = {}
    termW = {}

    uniqueRatings =  set(rating.values())
    
    for x in uniqueRatings:
        count = 0
            
        for doc,rate in rating.items():
            if rate == x and doc in trainDoc:
                count = count + 1
                docs_rating[rate] = count
        
    
        for term,ID in w.keys():
            if ID in trainDoc and rating[ID] == x:
                if term not in termW.keys():
                   termW[term] = w[(term,ID)]
                   
                else:
                   termW[term] = termW[term] + w[(term,ID)]
            protoVec[x] = termW
            
        termW = {};
        
    
    dict_pNorm = {} #the normalization factors for the protovectors
    
    for x in protoVec.keys():
        sum = 0
        for term,wi in protoVec[x].items():
            sum = sum + wi**2
        dict_pNorm[x] = sum;
    
    rocSim = {}
    
    for doc in testDoc:
        value = {}
        for x in protoVec.keys(): 
            sum = 0
            for term,wi in protoVec[x].items():
                if (term,doc) in w.keys():
                    if sim == 'cosine':
                        sum = sum + w[(term,doc)]*protoVec[x][term]
                    elif sim == 'dice':
                        sum = sum + 2*w[(term,doc)]*protoVec[x][term]
            if dict_pNorm[x] != 0:
                if sim == 'cosine':
                    value[x] =  sum/(dict_pNorm[x]*n[doc])**0.5
                if sim == 'dice':
                    value[x] =  sum/(dict_pNorm[x]+n[doc])

            else:
                value[x] = 0 
            rocSim[doc] = value;
        
    predActRating = {}
    
    for doc in rocSim.keys():
        predActRating[doc] = [max(rocSim[doc], key=rocSim[doc].get), rating[doc]]
    
    #Computing accuracy
    count = 0
    
    for val in predActRating.values():
        if val[0] == val[1]:
            count = count + 1
    
    accRoc = count/len(testDoc)
    
    print(accRoc)



####VERSION 2: CATEGORIZING DOCUMENTS USING K-NN WITH TFXIDF TERM WEIGHTS AND COSINE SIMILARITY

def knn(k = 5, w=tf_idf, sim = 'cosine'):
    '''This function predicts the category of the testing documents by using the k-nearest
    neighbors method. It takes the number of nearest neighbours to consider (k), the term 
    weights (tf, tf_idf, or binary), and the similarity method (cosine or dice) as the parameters.
    it calculates the similarity between each of the testing documents and the training documents
    and finds the k nearest neighbours. Based on the category majority of the k neighbours fall into
    the category of the test document is ascertained. In thecase of a tie, the higher rating is picked
    as the category. After the categories are predicted, it compares these vales with the actual categories
    of the testing documents to give an accuracy value. 
    '''
    global allSim
    global accKnn
    global ratingCounts
    global knnpredRatings
    
    if w == tf_idf:
        n = norm_tf_idf
    if w == tf:
        n = norm_tf
    if w == binary:
        n = norm_binary
    
    allSim = {}
    
    for testdoc in testDoc:
        allSim[testdoc] = {}
        for word, traindoc in w.keys():
            if (word, testdoc) in w.keys():
                if traindoc != testdoc:
                    if sim == 'cosine':  
                        if traindoc in allSim[testdoc].keys():
                            allSim[testdoc][traindoc] = allSim[testdoc][traindoc] + w[(word, testdoc)]*w[(word, traindoc)]
                        else:
                            allSim[testdoc][traindoc] = w[(word, testdoc)]*w[(word, traindoc)]
                    elif sim == 'dice':
                        if traindoc in allSim[testdoc].keys():
                            allSim[testdoc][traindoc] = allSim[testdoc][traindoc] + 2*w[(word, testdoc)]*w[(word, traindoc)]
                        else:
                            allSim[testdoc][traindoc] = 2*w[(word, testdoc)]*w[(word, traindoc)]
    
    for testdoc in allSim.keys(): ##normalizing to find the cosine similarity
        for traindoc in allSim[testdoc].keys():
            if sim == 'cosine':
                allSim[testdoc][traindoc] = allSim[testdoc][traindoc]/(n[testdoc]*n[traindoc])**0.5   
            elif sim == 'dice':
                allSim[testdoc][traindoc] = allSim[testdoc][traindoc]/(n[testdoc]+n[traindoc]) 
                
    #subsetting to only the nearest neighbours
    for testdoc in allSim.keys():
        allSim[testdoc] = sorted(allSim[testdoc].items(),key = operator.itemgetter(1), reverse = True)[0:k]

    #based on the nearest neighbours, we predict the rating, in a tie, we pick the highest rating
    knnRatings = {}
    
    for testdoc in allSim.keys():
        knnRatings[testdoc] = {}
        for traindoc in allSim[testdoc]:
            knnRatings[testdoc][traindoc[0]] = rating[traindoc[0]]
    
    ratingCounts = {}    
    for testdoc in knnRatings.keys():
        ratingCounts[testdoc] = {}
        for rate in set(knnRatings[testdoc].values()):
            ratingCounts[testdoc][rate] = sum(1 for x in knnRatings[testdoc].values() if x == rate)    
    
    knnpredRatings = {}

    for testdoc in ratingCounts.keys():
        if len(ratingCounts[testdoc]) == 0:
            continue
        knnpredRatings[testdoc] = sorted(ratingCounts[testdoc].items(),key = operator.itemgetter(1,0), reverse = True)[0]


    #Computing accuracy
    count = 0
    
    for doc in knnpredRatings.keys():
        if knnpredRatings[doc][0] == rating[doc]:
            count = count + 1
    
    accKnn = count/len(testDoc)
    
    print(accKnn)