#!/usr/bin/env python3
import sys
import os
from os.path import join,exists,dirname
import random
from datetime import datetime
import pickle
import time
import argparse

import torch.nn as nn
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from joint_ae_uda import JointLearnerModel, get_ae_inds, log
from pre import XML2arrayRAW, GetTopNMI

domains = ['books', 'dvd', 'electronics', 'kitchen']
parser = argparse.ArgumentParser(description='PyTorch joint domain adaptation neural network debugger')
parser.add_argument('-s', '--source', required=True, choices=domains)
parser.add_argument('-t', '--target', required=True, choices=domains)
parser.add_argument('-m', '--method', default='freq', choices=['freq', 'mi', 'ae', 'mi-ae', 'freq-ae'])

def main(args):
    if not torch.cuda.is_available():
        sys.stderr.write('WARNING: CUDA is not available... this may run very slowly!')


    args = parser.parse_args()

    src = args.source
    dest = args.target
    log("Running system on source=%s and target=%s" % (src, dest))
    
    filename = src + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename)): 
        sys.stderr.write('Data directory does not yet exist; other script should create it and then run this one\n')
        sys.exit(-1)
        #gets the dev set and train set for sentiment classification
        #train, train_labels, test, test_labels = extract_and_split("data/"+src+"/negative.parsed","data/"+src+"/positive.parsed")
        #target_train, target_train_labels, target_test, target_test_labels = extract_and_split("data/"+dest+"/negative.parsed","data/"+dest+"/positive.parsed")
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

    unlabeled_only,source_only,target_un=XML2arrayRAW("data/"+src+"/"+src+"UN.txt","data/"+dest+"/"+dest+"UN.txt")
    source_all=source_only+train
    un_count = 40

    lbl_num = 1000
    dest_test, _,_ = XML2arrayRAW("data/"+dest+"/negative.parsed","data/"+dest+"/positive.parsed")
    dest_test_labels= [0]*lbl_num+[1]*lbl_num

    unlabeled=source_all+target_un

    encoder_ae = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=un_count, binary=True)

    X_unlabeled_ae = encoder_ae.fit_transform(unlabeled)
    X_source_ae = encoder_ae.transform(train)
    X_allsource_ae = encoder_ae.transform(source_all)
    X_target_ae = encoder_ae.transform(dest_test)

    model = torch.load('best_model.pt')
    model.eval()
    num_pivots = model.rep_predictor.weight.shape[1]
    pivot_min_count = 10

    device='cuda' if torch.cuda.is_available() else 'cpu'

    ae_input_inds, ae_output_inds = get_ae_inds(args.method, X_source_ae, X_target_ae, X_unlabeled_ae, train_labels, num_pivots, X_allsource_ae, pivot_min_count)

    source_X_ae = torch.FloatTensor(X_source_ae[:,ae_input_inds].toarray()).to(device)
    target_X_ae = torch.FloatTensor(X_target_ae[:,ae_input_inds].toarray()).to(device)

    source_test_predict_raw,_,source_reps,_,_ = model(None, source_X_ae)
    source_reps_mat = source_reps.data.cpu().numpy()
    norm = np.linalg.norm(source_reps_mat, axis=1)
    source_reps_mat = source_reps_mat / norm[:, np.newaxis]

    target_test_predict_raw,_,target_reps,_,_ = model(None, target_X_ae)
    target_preds = torch.sigmoid(target_test_predict_raw) * 2 - 1

    target_reps_mat = target_reps.data.cpu().numpy()
    norm = np.linalg.norm(target_reps_mat, axis=1)
    target_reps_mat = target_reps_mat / norm[:, np.newaxis]

    # Create a similarity matrix by multiplying these 2 together -- dot products of all representations
    # with all similarities for source instance 0 in row 0, and similarities for target instance 0 in column 0
    sim_matrix = np.matmul(source_reps_mat, target_reps_mat.transpose())

    # we want the target instance whose closest similarity is furthest.
    # so take the column-wise max, to fnid the closest source instance to every column instance
    # Then take the min of that vector, to find the target instance that is furthest from a source instance
    most_sim = sim_matrix.max(0)

    sim_inds = np.argsort(most_sim)
    gold_labels = torch.FloatTensor(dest_test_labels).to(device) * 2 - 1
    err = (gold_labels * target_preds[:,0]).detach().cpu().numpy()
    margin_inds = np.argsort(abs(err))

    egs = 100
    for ind in margin_inds:
        if err[ind] < 0:
            print('######## Error %d (ind %d) was %f from the margin with gold label %d ' % (100-egs, ind, err[ind], dest_test_labels[ind]))
            print(dest_test[ind])
            print('########################################################')
            egs -=1
        if egs <= 0:
            break

    err_inds = np.argsort(err)

    
    

    

if __name__ == '__main__':
    main(sys.argv[1:])
