import xml.etree.ElementTree as ET
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import pickle
import os

def XML2arrayRAW(neg_path, pos_path):
    reviews = []
    negReviews = []
    posReviews = []

    neg_tree = ET.parse(neg_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        reviews.append(rev.text)
        negReviews.append(rev.text)



    pos_tree = ET.parse(pos_path)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review'):
        reviews.append(rev.text)
        posReviews.append(rev.text)

    return reviews,negReviews,posReviews



def split_data_balanced(reviews,dataSize,testSize):
    test_data_neg = random.sample(range(0, dataSize), testSize)
    test_data_pos = random.sample(range(dataSize, 2*dataSize), testSize)
    random_array = np.concatenate((test_data_neg, test_data_pos))
    train = []
    test = []
    test_labels = []
    train_labels = []
    for i in range(0, 2*dataSize):
        if i in random_array:
            test.append(reviews[i])
            labels = 0 if i < dataSize else 1
            test_labels.append(labels)
        else:
            train.append(reviews[i])
            labels = 0 if i < dataSize else 1
            train_labels.append(labels)
    return train, train_labels, test, test_labels

def extract_and_split(neg_path, pos_path):
    reviews,n,p = XML2arrayRAW(neg_path, pos_path)
    train, train_labels, test, test_labels = split_data_balanced(reviews,1000,200)
    return train, train_labels, test, test_labels


def GetTopNMI(n,X,labels):
    MI = []
    length = X.shape[1]


    for i in range(length):
        temp=mutual_info_score(X[:, i], labels)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI


def getCounts(X,i):

    return (sum(X[:,i]))

def preproc(pivot_num,pivot_min_st,src,dest, pivot_method='mi', pivots=None):

    pivotsCounts= []
    unlabeled = []
    names = []
    #if the split is not already exists, extract it, otherwise, load an existing one.
    filename = src + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename)):
        #gets the dev set and train set for sentiment classification
        train, train_labels, test, test_labels = extract_and_split("data/"+src+"/negative.parsed","data/"+src+"/positive.parsed")
        target_train, target_train_labels, target_test, target_test_labels = extract_and_split("data/"+dest+"/negative.parsed","data/"+dest+"/positive.parsed")
    #loads an existing split
    else:
        with open(src + "_to_" + dest + "/split/train", 'rb') as f:
            train = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test", 'rb') as f:
            test = pickle.load(f)
        with open(src + "_to_" + dest + "/split/train_labels", 'rb') as f:
            train_labels = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test_labels", 'rb') as f:
            test_labels = pickle.load(f)
        with open(src + "_to_" + dest + "/split/target_train", 'rb') as f:
            target_train = pickle.load(f)
        with open(src + "_to_" + dest + "/split/target_test", 'rb') as f:
            target_test = pickle.load(f)
        with open(src + "_to_" + dest + "/split/target_train_labels", 'rb') as f:
            target_train_labels = pickle.load(f)
        with open(src + "_to_" + dest + "/split/target_test_labels", 'rb') as f:
            target_test_labels = pickle.load(f)
       


    # sets x train matrix for classification
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=5,binary=True)
    X_2_train = bigram_vectorizer.fit_transform(train).toarray()

    # gets all the train and test for pivot classification
    unlabeled_only,source,target=XML2arrayRAW("data/"+src+"/"+src+"UN.txt","data/"+dest+"/"+dest+"UN.txt")
    source=source+train
    src_count = 20
    un_count = 40


    unlabeled=source+target

    bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=un_count, binary=True)
    X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()

    # at this point the 'source' variable is unlabeled + labeled data from the source domain
    bigram_vectorizer_source = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=src_count, binary=True)
    X_2_train_source = bigram_vectorizer_source .fit_transform(source).toarray()
   
    # target is the unlabeled data from the target domain (i.e. this preproc does not make use of the set of target data for which there is labels, even by ignoring the labels
    bigram_vectorizer_target = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20, binary=True)
    X_2_train_target = bigram_vectorizer_target.fit_transform(target).toarray()

    X_2_train_target_main_encoding = bigram_vectorizer.transform(target_train).toarray()
    
    #gets a sorted list of pivots with respect to the MI with the label
    MIsorted,RMI=GetTopNMI(2000,X_2_train,train_labels)
    MIsorted_target,RMI_target=GetTopNMI(2000, X_2_train_target_main_encoding, target_train_labels)
    
    RMI_joint_oracle = np.array(RMI) * np.array(RMI_target)
    MIsorted_joint_oracle = sorted(range(len(RMI_joint_oracle)), key=lambda i: RMI_joint_oracle[i])

    #Get a list of 
    domain_target = np.ones( len(unlabeled) )
    domain_target[:len(source)] = 0
    MIdomsorted,DMI = GetTopNMI(len(unlabeled), X_2_train_unlabeled, domain_target)
    dom_spec_feat_inds = np.where(np.array(DMI) > 0.01)[0]
    dom_spec_feat_names = []
    for ind in dom_spec_feat_inds:
        dom_spec_feat_names.append( bigram_vectorizer_unlabeled.get_feature_names()[ind] )
    
    if pivot_method == 'mi' or pivot_method == 'mi-domfilter':
        MIsorted.reverse()
        pivots = MIsorted
    elif pivot_method == 'random':
        random_indices = np.arange(X_2_train.shape[1])
        random.shuffle(random_indices)
        pivots = random_indices
    elif pivot_method == 'list':
        pivot_inds = []
        for pivot in pivots:
            if pivot in bigram_vectorizer.get_feature_names():
                pivot_inds.append( bigram_vectorizer.get_feature_names().index(pivot) )
        pivots = pivot_inds
        print('%d pivots remain after filtering input pivots through bigram vectorizer' % (len(pivots)))
    elif pivot_method == 'learned':
        import pivot_learner_model
        # FIXME - uncomment this when i start using it, but for debugging its faster without
        X_2_train_unlabeled_un_encoded = None #bigram_vectorizer_unlabeled.transform(unlabeled_only).toarray()
        X_2_train_source_un_encoded = bigram_vectorizer_unlabeled.transform(train).toarray()
        X_2_train_target_un_encoded = bigram_vectorizer_unlabeled.transform(target_train).toarray()

        pivots = pivot_learner_model.get_pivots(X_2_train_unlabeled_un_encoded, X_2_train_source_un_encoded, train_labels, X_2_train_target_un_encoded)
    elif pivot_method == 'oracle':
        # FIXME copy above
        #pivots = pivot_learner_model.get_pivots(X_2_train_unlabeled_trainencoded, X_2_train, train_labels, X_2_train_target_trainencoded, target_train_labels)
        # raise Exception('Not implemented yet: %s' % (pivot_method) )
        MIsorted_joint_oracle.reverse()
        pivots = MIsorted_joint_oracle
    else:
        raise Exception('No known pivot selection method supplied: %s' % (pivot_method) )
        
    c=0
    i=0

    while (c<pivot_num):
        name= bigram_vectorizer.get_feature_names()[pivots[i]]
        

        s_count = getCounts(X_2_train_source,bigram_vectorizer_source.get_feature_names().index(name)) if name in bigram_vectorizer_source.get_feature_names() else 0
        t_count = getCounts(X_2_train_target, bigram_vectorizer_target.get_feature_names().index(name)) if name in bigram_vectorizer_target.get_feature_names() else 0
        #pivot must meet 2 conditions, to have high MI with the label and appear at least pivot_min_st times in the source and target domains
        if(s_count>=pivot_min_st and t_count>=pivot_min_st):
            if pivot_method == 'mi-domfilter' and name in dom_spec_feat_names:
                print("Ignoring feature %s because it has high MI with the domain" % (name) )
            else:
                names.append(name)
                pivotsCounts.append(bigram_vectorizer_unlabeled.get_feature_names().index(name))
                c+=1
                print("feature %d is '%s' its MI is %f. Count in source: %d, and in target: %d" % (c, name, RMI[MIsorted[i]], s_count, t_count))
        i+=1


    #takes out fifth of the training data for validation(with respect to the represantation learning task)
    source_valid = len(source)//5
    target_valod = len(target)//5
    c=0
    y = X_2_train_unlabeled[:,pivotsCounts]
    x =np.delete(X_2_train_unlabeled, pivotsCounts, 1)  # delete second column of C
    x_valid=np.concatenate((x[:source_valid][:], x[-target_valod:][:]), axis=0)

    x = x[source_valid:-target_valod][:]


    #we take fifth of the source examples and fifth of the target examples to keep the same ratio in validation
    y_valid = np.concatenate((y[:source_valid][:], y[-target_valod:][:]), axis=0)

    y = y[source_valid:-target_valod][:]
    filename = src+"_to_"+dest+"/"+"pivot_names/pivot_names_"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st)
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))


    with open(filename, 'wb') as f:
        pickle.dump(names, f)
    filename = src + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        with open(src + "_to_" + dest + "/split/train", 'wb') as f:
            pickle.dump(train, f)
        with open(src + "_to_" + dest + "/split/test", 'wb') as f:
            pickle.dump(test, f)
        with open(src + "_to_" + dest + "/split/train_labels", 'wb') as f:
            pickle.dump(train_labels, f)
        with open(src + "_to_" + dest + "/split/test_labels", 'wb') as f:
            pickle.dump(test_labels, f)
        with open(src + "_to_" + dest + "/split/target_train", 'wb') as f:
            pickle.dump(target_train, f)
        with open(src + "_to_" + dest + "/split/target_test", 'wb') as f:
            pickle.dump(target_test, f)
        with open(src + "_to_" + dest + "/split/target_train_labels", 'wb') as f:
            pickle.dump(target_train_labels, f)
        with open(src + "_to_" + dest + "/split/target_test_labels", 'wb') as f:
            pickle.dump(target_test_labels, f)
    filename = src+"_to_"+dest+"/"+"pivotsCounts/"+"pivotsCounts"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st)
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    with open(src+"_to_"+dest+"/"+"pivotsCounts/"+"pivotsCounts"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st), 'wb') as f:
        pickle.dump(pivotsCounts, f)


    #finally, we return the training and validation data, there is not test data since we do not care about the test in the representation learning task
    return x,y,x_valid,y_valid,x.shape[1]

