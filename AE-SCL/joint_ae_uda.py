#!/usr/bin/env python3
import sys
import os
from os.path import join,exists,dirname
import random
from datetime import datetime
import pickle
import time

import torch.nn as nn
import torch.optim as optim
import torch
from torch import sigmoid
from torch.nn.functional import relu
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from pre import XML2arrayRAW

class JointLearnerModel(nn.Module):

    def __init__(self, input_features, non_pivot_candidates, num_pivot_candidates, pivot_hidden_nodes=100):
        super(JointLearnerModel, self).__init__()

        # The task net takes a concatenated input vector + predicted pivot vector and maps it to a prediction for the task
        self.task_net = nn.Sequential()
        num_features = input_features + pivot_hidden_nodes
        
        # task_classifier maps from a feature representation to a task prediction
        self.task_classifier = nn.Sequential()
        self.task_classifier.add_module('task_binary', nn.Linear(num_features, 1))
        #self.task_classifier.add_module('task_sigmoid', nn.Sigmoid())
        
        # domain classifier maps from a feature representation to a domain prediction
        #self.pivot_ae = nn.Sequential()
        self.rep_projector = nn.Linear(non_pivot_candidates, pivot_hidden_nodes)
        self.rep_predictor = nn.Linear(pivot_hidden_nodes, num_pivot_candidates)
        
    def forward(self, full_input, pivot_input):

        # Get predictions for all pivot candidates:
        pivot_rep = relu(self.rep_projector(pivot_input))
        pivot_pred = self.rep_predictor(pivot_rep)

        task_input = torch.cat( (full_input, pivot_rep), dim=1 )
        
        # Get task prediction
        task_prediction = self.task_classifier(task_input)
        # oracle_prediction = self.oracle_classifier(task_input)

        return task_prediction, pivot_pred

def get_shuffled(X, y=None):
    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)
    shuffled_X = X[inds, :]
    if y is None:
        shuffled_y = None
    else:
        shuffled_y = y[inds]
    return shuffled_X, shuffled_y

def log(msg):
    sys.stdout.write('%s\n' % msg)
    sys.stdout.flush()

def train_model(X_train_source, y_train_source, X_train_target, pivot_candidate_inds, non_pivot_candidate_inds, X_train_unlabeled=None, y_train_target=None, X_test_source=None, y_test_source=None):
    assert X_train_source.shape[1] == X_train_target.shape[1], "Source and target training data do not have the same number of features!"

    device='cuda' if torch.cuda.is_available() else 'cpu'
 
    epochs = 20
    recon_weight = 1.0
    # oracle_weight = 1.0
    max_batch_size = 50
    pivot_hidden_nodes = 500
    weight_decay = 0.01
 
    if y_train_target is None:
        log('Proceeding in standard semi-supervised pivot-learning mode')
    else:
        log('Proceeding in oracle mode (using target labels to jointly train pivot learner')
    

    log('There are %d candidate pivot features that meet source and target frequency requirements and %d pivot predictors' % (len(pivot_candidate_inds), len(non_pivot_candidate_inds)))
    
    num_source_instances, num_features = X_train_source.shape
    num_target_instances = X_train_target.shape[0]
    if num_source_instances > num_target_instances:
        source_batch_size = max_batch_size
        num_batches = (num_source_instances // max_batch_size)
        target_batch_size = (num_target_instances // num_batches) 
    else:
        target_batch_size = max_batch_size
        num_batches = (num_target_instances // max_batch_size)
        source_batch_size = (num_source_instances // num_batches)

    if not X_train_unlabeled is None:
        num_unlabeled_instances = X_train_unlabeled.shape[0]
        if num_unlabeled_instances > num_source_instances:
            un_batch_size = num_unlabeled_instances // num_batches
            log("Unlabeled data will be processed in batches of size %d" % (un_batch_size))
        else:
            raise Exception("ERROR: There are too few unlabeled instances. Is something wrong?\n")

    model = JointLearnerModel(num_features, len(non_pivot_candidate_inds), len(pivot_candidate_inds), pivot_hidden_nodes=pivot_hidden_nodes).to(device)
    task_lossfn = nn.BCEWithLogitsLoss().to(device)
    recon_lossfn = nn.BCEWithLogitsLoss().to(device)

    opt = optim.Adam(model.parameters(), weight_decay=weight_decay) 
    best_valid_f1 = 0

    for epoch in range(epochs):
        source_batch_ind = 0
        target_batch_ind = 0
        un_batch_ind = 0
        epoch_loss = 0
        tps = 0             # num for P/R
        true_labels = 0     # denom for recall
        true_preds = 0      # denom for prec
        correct_preds = 0   # For accuracy

        source_X, source_y = get_shuffled(X_train_source, y_train_source)
        target_X, target_y = get_shuffled(X_train_target, y_train_target)
        unlabeled_X,_ = get_shuffled(X_train_unlabeled)

        for batch in range(num_batches):
            model.zero_grad()

            # Pass it source examples and compute task loss and pivot reconstruction loss:
            batch_source_X = torch.FloatTensor(source_X[source_batch_ind:source_batch_ind+source_batch_size, :]).to(device)
            non_pivot_inputs = torch.FloatTensor(source_X[source_batch_ind:source_batch_ind+source_batch_size,non_pivot_candidate_inds]).to(device)
            pivot_labels = torch.FloatTensor(source_X[source_batch_ind:source_batch_ind+source_batch_size,pivot_candidate_inds]).to(device)
            task_pred,pivot_pred = model(batch_source_X, non_pivot_inputs)
            batch_source_y = torch.FloatTensor(source_y[source_batch_ind:source_batch_ind+source_batch_size]).to(device).unsqueeze(1)
            task_loss = task_lossfn(task_pred, batch_source_y)
           
            # since we don't have a sigmoid in our network's task output (it is part of the loss function for numerical stability), we need to manually apply the sigmoid if we want to do some standard acc/p/r/f calculations.
            task_bin_pred = np.round(sigmoid(task_pred).data.cpu().numpy())[:,0]
            true_preds += task_bin_pred.sum().item()
            true_labels += batch_source_y.sum().item()
            tps += (task_bin_pred * batch_source_y[:,0]).sum().item()
            correct_preds += (task_bin_pred == batch_source_y[:,0]).sum().item()

            source_recon_loss = recon_lossfn(pivot_pred, pivot_labels)

            # pass it target examples and compute reconstruction loss:
            batch_target_X = torch.FloatTensor(target_X[target_batch_ind:target_batch_ind+target_batch_size, :]).to(device)
            non_pivot_inputs = torch.FloatTensor(target_X[target_batch_ind:target_batch_ind+target_batch_size,non_pivot_candidate_inds]).to(device)
            pivot_labels = torch.FloatTensor(target_X[target_batch_ind:target_batch_ind+target_batch_size,pivot_candidate_inds]).to(device)
            target_task_pred,pivot_pred = model(batch_target_X, non_pivot_inputs)
            # No task loss because we don't have target labels

            target_recon_loss = recon_lossfn(pivot_pred, pivot_labels)

            # do representation learning on the unlabeled instances
            # batch_unlabeled_X = unlabeled_X[un_batch_ind:un_batch_ind+un_batch_size, :]
            # if un_batch_size > max_batch_size:
            num_sub_batches = 1 + (un_batch_size // max_batch_size)
            sub_batch_start_ind = 0
            unlabeled_recon_loss = 0.0
            for sub_batch in range(num_sub_batches):
                sub_batch_size = min(max_batch_size, un_batch_size - sub_batch*max_batch_size )
                if sub_batch_size <= 0:
                    print('Found an edge case where sub_batch_size<=0 with un_batch_size=%d' % (un_batch_size))
                    break
                #if sub_batch_size < max_batch_size:
                    #print('Batch %d has size %d' % (sub_batch, sub_batch_size))

                sub_batch_unlabeled_X = torch.FloatTensor(unlabeled_X[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size, :]).to(device)
                non_pivot_inputs = torch.FloatTensor(unlabeled_X[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size,non_pivot_candidate_inds]).to(device)
                pivot_labels = torch.FloatTensor(unlabeled_X[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size, pivot_candidate_inds]).to(device)
                _, pivot_pred = model(sub_batch_unlabeled_X, non_pivot_inputs)
                unlabeled_recon_loss += recon_lossfn(pivot_pred, pivot_labels)
                sub_batch_start_ind += max_batch_size


            # Compute the total loss and step the optimizer in the right direction:
            total_loss = (task_loss + 
                         recon_weight * (source_recon_loss + target_recon_loss + unlabeled_recon_loss))
            epoch_loss += total_loss.item()
            total_loss.backward()
     
            opt.step()
            
            source_batch_ind += source_batch_size
            target_batch_ind += target_batch_size
            un_batch_ind += un_batch_size

        # Print out some useful info: losses, weights of pivot filter, accuracy
        acc = correct_preds / source_batch_ind
        prec = tps / true_preds
        rec = tps / true_labels
        f1 = 2 * prec * rec / (prec + rec)
        log("Epoch %d finished: loss=%f" % (epoch, epoch_loss) )
        log("  Training accuracy=%f, p=%f, rec=%f, f1=%f" % (acc, prec, rec, f1))

        if not X_test_source is None:
            test_X = torch.FloatTensor(X_test_source).to(device)
            test_np_input = torch.FloatTensor(X_test_source[:, non_pivot_candidate_inds]).to(device)
            # test_y = torch.FloatTensor(y_test_source).to(device)
            test_preds = np.round(sigmoid(model(test_X, test_np_input)[0]).data.cpu().numpy())[:,0]
            correct_preds = (y_test_source == test_preds).sum()
            acc = correct_preds / len(y_test_source)

            tps = (test_preds * y_test_source).sum().item()
            test_true_labels = y_test_source.sum().item()
            test_true_preds = test_preds.sum()
            rec = tps / test_true_labels
            prec = tps / test_true_preds
            f1 = 2 * rec * prec / (rec + prec)
            log("  Validation accuracy=%f, p=%f, rec=%f, f1=%f" % (acc, prec, rec, f1))

            if f1 > best_valid_f1:
                sys.stderr.write('Writing model at epoch %d\n' % (epoch))
                sys.stderr.flush()
                best_valid_f1 = f1
                torch.save(model, 'best_model.pt')


def main(args):
    if not torch.cuda.is_available():
        sys.stderr.write('WARNING: CUDA is not available... this may run very slowly!')
        
    domain = [] 
    domain.append("books")
    domain.append("dvd")
    domain.append("electronics")
    domain.append("kitchen")

    source_ind = 2
    target_ind = 1

    log("Running system on source=%s and target=%s" % (domain[source_ind], domain[target_ind]))

    un_count = 40
    src = domain[source_ind]
    dest = domain[target_ind]

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

    # gets all the train and test for pivot classification
    unlabeled_only,source,target=XML2arrayRAW("data/"+src+"/"+src+"UN.txt","data/"+dest+"/"+dest+"UN.txt")
    source=source+train
    un_count = 40

    lbl_num = 1000
    dest_test, _,_ = XML2arrayRAW("data/"+dest+"/negative.parsed","data/"+dest+"/positive.parsed")
    dest_test_labels= [0]*lbl_num+[1]*lbl_num

    unlabeled=source+target

    bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=un_count, binary=True)
    # This array isn't used anywhere but it's used to train the vectorizer:
    X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()

    X_2_train_unlabeled_un_encoded = bigram_vectorizer_unlabeled.transform(unlabeled_only).toarray()
    X_2_train_source_un_encoded = bigram_vectorizer_unlabeled.transform(train).toarray()
    X_2_train_target_un_encoded = bigram_vectorizer_unlabeled.transform(dest_test).toarray()
    X_2_test_source_un_encoded = bigram_vectorizer_unlabeled.transform(test).toarray()

    source_cands = np.where(X_2_train_source_un_encoded.sum(0) > 10)[0]
    target_cands = np.where(X_2_train_target_un_encoded.sum(0) > 10)[0]
    # pivot candidates are those that meet frequency cutoff in both domains train data:
    pivot_candidate_inds = np.intersect1d(source_cands, target_cands)
    # non-pivot candidates are the set difference - those that didn't meet the frequency cutoff in both domains:
    non_pivot_candidate_inds = np.setdiff1d(range(X_2_train_source_un_encoded.shape[1]), pivot_candidate_inds)

    train_model(X_2_train_source_un_encoded, 
                np.array(train_labels), 
                X_2_train_target_un_encoded,
                pivot_candidate_inds,
                non_pivot_candidate_inds,
                X_2_train_unlabeled_un_encoded,
                y_train_target=None,  # If we come up with oracle mode this can be non-None
                X_test_source=X_2_test_source_un_encoded, 
                y_test_source=np.array(test_labels))

    best_model = torch.load('best_model.pt')

    device='cuda' if torch.cuda.is_available() else 'cpu'

    X_2_test_target_un_encoded = bigram_vectorizer_unlabeled.transform(dest_test).toarray()
    X_2_test_target_non_pivot = X_2_test_target_un_encoded[:,non_pivot_candidate_inds]

    target_X = torch.FloatTensor(X_2_test_target_un_encoded).to(device)
    target_y = np.array(dest_test_labels)
    target_X_nonpivot = torch.FloatTensor(X_2_test_target_non_pivot).to(device)

    target_test_predict_raw,_ = best_model(target_X, target_X_nonpivot)
    target_test_predict = np.round(sigmoid(target_test_predict_raw).data.cpu().numpy())[:,0]

    correct_preds = (target_y == target_test_predict).sum()
    acc = correct_preds / len(target_y)

    tps = (target_test_predict * target_y).sum().item()
    test_true_labels = target_y.sum()
    test_true_preds = target_test_predict.sum()
    rec = tps / test_true_labels
    prec = tps / test_true_preds
    f1 = 2 * rec * prec / (rec + prec)
    log("  Target accuracy=%f, p=%f, rec=%f, f1=%f" % (acc, prec, rec, f1))



if __name__ == '__main__':
    main(sys.argv[1:])

