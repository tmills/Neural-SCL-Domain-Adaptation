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
import torch.optim as optim
import torch
from torch import sigmoid
from torch.nn.functional import relu
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from pre import XML2arrayRAW, GetTopNMI

class JointLearnerModel(nn.Module):

    def __init__(self, input_features, non_pivot_candidates, num_pivot_candidates, pivot_hidden_nodes=100):
        super(JointLearnerModel, self).__init__()

        # The task net takes a concatenated input vector + predicted pivot vector and maps it to a prediction for the task
        self.task_net = nn.Sequential()
        num_features = input_features + pivot_hidden_nodes
        
        # task_classifier maps from a feature representation to a task prediction
        self.task_classifier = nn.Linear(num_features,1)
        
        # domain classifier maps from a feature representation to a domain prediction
        #self.pivot_ae = nn.Sequential()
        self.rep_projector = nn.Linear(non_pivot_candidates, pivot_hidden_nodes)
        self.rep_predictor = nn.Linear(pivot_hidden_nodes, num_pivot_candidates)
        
    def forward(self, full_input, pivot_input):

        # Get predictions for all pivot candidates:
        pivot_rep = sigmoid(self.rep_projector(pivot_input))
        pivot_pred = self.rep_predictor(pivot_rep)

        task_input = torch.cat( (full_input, pivot_rep), dim=1 )
        
        # Get task prediction
        task_prediction = self.task_classifier(task_input)
        # oracle_prediction = self.oracle_classifier(task_input)

        return task_prediction, pivot_pred, pivot_rep

def get_shuffled(X_feats, X_ae=None, y=None):
    inds = np.arange(X_feats.shape[0])
    np.random.shuffle(inds)
    shuffled_X_feats = X_feats[inds, :]

    if X_ae is None:
        shuffled_X_ae = None
    else:
        shuffled_X_ae = X_ae[inds, :]

    if y is None:
        shuffled_y = None
    else:
        shuffled_y = y[inds]
    return shuffled_X_feats, shuffled_X_ae, shuffled_y

def log(msg):
    sys.stdout.write('%s\n' % msg)
    sys.stdout.flush()

def train_model(X_source_feats, X_source_ae, y_source, X_target_feats, X_target_ae, ae_input_inds, ae_output_inds, y_target=None, X_unlabeled_feats=None, X_unlabeled_ae=None,  X_source_valid_feats=None, X_source_valid_ae=None, y_source_valid=None):
    assert X_source_feats.shape[1] == X_target_feats.shape[1], "Source and target training data do not have the same number of features!"
    assert X_source_feats.shape[1] == X_unlabeled_feats.shape[1], "Source and unlabeled training data do not have the same number of features!"

    device='cuda' if torch.cuda.is_available() else 'cpu'
 
    epochs = 25
    recon_weight = 100.0
    l2_weight = 0.0 #1
    # oracle_weight = 1.0
    max_batch_size = 50
    pivot_hidden_nodes = 500
    weight_decay = 0.0000 #1
    lr = 0.001  # adam default is 0.001
 
    log('Proceeding in standard semi-supervised pivot-learning mode')
    
    log('There are %d auto-encoder output features that meet source and target frequency requirements and %d predictors' % (len(ae_output_inds), len(ae_input_inds)))
    
    num_source_instances, num_features = X_source_feats.shape
    num_target_instances = X_target_feats.shape[0]
    if num_source_instances > num_target_instances:
        source_batch_size = max_batch_size
        num_batches = (num_source_instances // max_batch_size)
        target_batch_size = (num_target_instances // num_batches) 
    else:
        target_batch_size = max_batch_size
        num_batches = (num_target_instances // max_batch_size)
        source_batch_size = (num_source_instances // num_batches)

    if not X_unlabeled_feats is None:
        assert X_unlabeled_ae is not None, 'If X_unlabeled_feats is present, X_unlabeled_ae must also be present'
        num_unlabeled_instances = X_unlabeled_feats.shape[0]
        if num_unlabeled_instances > num_source_instances:
            un_batch_size = num_unlabeled_instances // num_batches
            log("Unlabeled data will be processed in batches of size %d" % (un_batch_size))
        else:
            raise Exception("ERROR: There are too few unlabeled instances. Is something wrong?\n")

    model = JointLearnerModel(num_features, len(ae_input_inds), len(ae_output_inds), pivot_hidden_nodes=pivot_hidden_nodes).to(device)
    task_lossfn = nn.BCEWithLogitsLoss().to(device)
    recon_lossfn = nn.BCEWithLogitsLoss().to(device)
    l2_lossfn = nn.MSELoss(reduction='sum').to(device)

    opt = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr) 
    best_valid_acc = 0

    for epoch in range(epochs):
        source_batch_ind = 0
        target_batch_ind = 0
        un_batch_ind = 0
        epoch_loss = 0
        tps = 0             # num for P/R
        true_labels = 0     # denom for recall
        true_preds = 0      # denom for prec
        correct_preds = 0   # For accuracy

        source_X, source_X_ae, source_y = get_shuffled(X_source_feats, X_source_ae, y_source)
        target_X, target_X_ae, target_y = get_shuffled(X_target_feats, X_target_ae, y_target)
        unlabeled_X, unlabeled_X_ae,_ = get_shuffled(X_unlabeled_feats, X_unlabeled_ae)

        for batch in range(num_batches):
            model.zero_grad()

            # Pass it source examples and compute task loss and pivot reconstruction loss:
            batch_source_X = torch.FloatTensor(source_X[source_batch_ind:source_batch_ind+source_batch_size, :].toarray()).to(device)

            ae_inputs = torch.FloatTensor(source_X_ae[source_batch_ind:source_batch_ind+source_batch_size,ae_input_inds].toarray()).to(device)

            task_pred,pivot_pred,_ = model(batch_source_X, ae_inputs)

            # Get task loss:
            batch_source_y = torch.FloatTensor(source_y[source_batch_ind:source_batch_ind+source_batch_size]).to(device).unsqueeze(1)
            task_loss = task_lossfn(task_pred, batch_source_y)

            # Get reconstruction loss:
            ae_outputs = torch.FloatTensor(source_X_ae[source_batch_ind:source_batch_ind+source_batch_size,ae_output_inds].toarray()).to(device)
            source_recon_loss = recon_lossfn(pivot_pred, ae_outputs)
           
            # since we don't have a sigmoid in our network's task output (it is part of the loss function for numerical stability), we need to manually apply the sigmoid if we want to do some standard acc/p/r/f calculations.
            task_bin_pred = np.round(sigmoid(task_pred).data.cpu().numpy())[:,0]
            true_preds += task_bin_pred.sum().item()
            true_labels += batch_source_y.sum().item()
            tps += (task_bin_pred * batch_source_y[:,0]).sum().item()
            correct_preds += (task_bin_pred == batch_source_y[:,0]).sum().item()


            # pass it target examples and compute reconstruction loss:
            batch_target_X = torch.FloatTensor(target_X[target_batch_ind:target_batch_ind+target_batch_size, :].toarray()).to(device)
            ae_inputs = torch.FloatTensor(target_X_ae[target_batch_ind:target_batch_ind+target_batch_size,ae_input_inds].toarray()).to(device)

            target_task_pred,pivot_pred,_ = model(batch_target_X, ae_inputs)

            # No task loss because we don't have target labels

            # Reconstruction loss:
            ae_outputs = torch.FloatTensor(target_X_ae[target_batch_ind:target_batch_ind+target_batch_size,ae_output_inds].toarray()).to(device)
            target_recon_loss = recon_lossfn(pivot_pred, ae_outputs)

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

                sub_batch_unlabeled_X = torch.FloatTensor(unlabeled_X[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size, :].toarray()).to(device)
                ae_inputs = torch.FloatTensor(unlabeled_X_ae[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size,ae_input_inds].toarray()).to(device)
                ae_outputs = torch.FloatTensor(unlabeled_X_ae[un_batch_ind+sub_batch_start_ind:un_batch_ind+sub_batch_start_ind+sub_batch_size, ae_output_inds].toarray()).to(device)
                _, pivot_pred,_ = model(sub_batch_unlabeled_X, ae_inputs)
                unlabeled_recon_loss += recon_lossfn(pivot_pred, ae_outputs)
                sub_batch_start_ind += max_batch_size


            # Compute the total loss and step the optimizer in the right direction:
            l2_loss = l2_lossfn( model.task_classifier.weight, torch.zeros_like(model.task_classifier.weight))
            total_loss = (task_loss + 
                         l2_weight * l2_loss +
                         recon_weight * (source_recon_loss + target_recon_loss + unlabeled_recon_loss))
            epoch_loss += total_loss.item()
            total_loss.backward()
     
            opt.step()
            
            source_batch_ind += source_batch_size
            target_batch_ind += target_batch_size
            un_batch_ind += un_batch_size

        # Print out some useful info: losses, weights of pivot filter, accuracy
        acc = correct_preds / source_batch_ind
        log("Epoch %d finished: loss=%f" % (epoch, epoch_loss) )
        log("  Training accuracy=%f" % (acc))

        if not X_source_valid_feats is None:
            test_X = torch.FloatTensor(X_source_valid_feats.toarray()).to(device)
            test_np_input = torch.FloatTensor(X_source_valid_ae[:, ae_input_inds].toarray()).to(device)
            # test_y = torch.FloatTensor(y_test_source).to(device)
            test_preds = np.round(sigmoid(model(test_X, test_np_input)[0]).data.cpu().numpy())[:,0]
            correct_preds = (y_source_valid == test_preds).sum()
            acc = correct_preds / len(y_source_valid)

            log("  Validation accuracy=%f" % (acc))

            if acc > best_valid_acc:
                sys.stderr.write('Writing model at epoch %d\n' % (epoch))
                sys.stderr.flush()
                best_valid_acc = acc
                torch.save(model, 'best_model.pt')

        del unlabeled_X

domains = ['books', 'dvd', 'electronics', 'kitchen']
parser = argparse.ArgumentParser(description='PyTorch joint domain adaptation neural network trainer')
parser.add_argument('-s', '--source', required=True, choices=domains)
parser.add_argument('-t', '--target', required=True, choices=domains)
parser.add_argument('-m', '--method', default='freq', choices=['freq', 'mi', 'ae'])
parser.add_argument('-e', '--eval', default='pt', choices=['pt', 'lr'])

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

    # gets all the train and test for pivot classification
    unlabeled_only,source_only,target=XML2arrayRAW("data/"+src+"/"+src+"UN.txt","data/"+dest+"/"+dest+"UN.txt")
    source_all=source_only+train
    un_count = 40
    num_pivots = 100   # Only used in configuration that uses MI to get pivots
    pivot_min_count = 10

    lbl_num = 1000
    dest_test, _,_ = XML2arrayRAW("data/"+dest+"/negative.parsed","data/"+dest+"/positive.parsed")
    dest_test_labels= [0]*lbl_num+[1]*lbl_num

    unlabeled=source_all+target

    encoder_feats = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20, binary=True)
    X_source_feats = encoder_feats.fit_transform(train)
    X_target_feats = encoder_feats.transform(dest_test)
    X_unlabeled_feats = encoder_feats.transform(unlabeled)
    X_source_valid_feats = encoder_feats.transform(test)

    encoder_ae = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=un_count, binary=True)
    X_unlabeled_ae = encoder_ae.fit_transform(unlabeled)

    X_unlabeledonly_ae = encoder_ae.transform(unlabeled_only)
    X_source_ae = encoder_ae.transform(train)
    X_allsource_ae = encoder_ae.transform(source_all)
    X_target_ae = encoder_ae.transform(dest_test)
    X_source_valid_ae = encoder_ae.transform(test)


    if args.method == 'freq':
        source_cands = np.where(X_source_ae.sum(0) > pivot_min_count)[1]
        target_cands = np.where(X_target_ae.sum(0) > pivot_min_count)[1]
        # pivot candidates are those that meet frequency cutoff in both domains train data:
        ae_output_inds = np.intersect1d(source_cands, target_cands)
        # non-pivot candidates are the set difference - those that didn't meet the frequency cutoff in both domains:
        ae_input_inds = np.setdiff1d(range(X_unlabeled_ae.shape[1]), ae_output_inds)
    elif args.method == 'ae':
        ae_input_inds = ae_output_inds = range(X_unlabeled_ae.shape[1])
    elif args.method == 'mi':
        # Run the sklearn mi feature selection:
        MIs, MI = GetTopNMI(2000, X_source_ae.toarray(), train_labels)
        MIs.reverse()
        ae_output_inds = []
        i=c=0
        while c < num_pivots:
            s_count = X_allsource_ae[:,i].sum()
            t_count = X_target_ae[:,i].sum()
            if s_count >= pivot_min_count and t_count >= pivot_min_count:
                ae_output_inds.append(MIs[i])
                c += 1
                print("feature %d is '%s' with mi %f" % (c, encoder_ae.get_feature_names()[MIs[i]], MI[MIs[i]]))
            i += 1


        ae_output_inds.sort()
        ae_input_inds = np.setdiff1d(range(X_unlabeled_ae.shape[1]), ae_output_inds)
        
    train_model(X_source_feats,
                X_source_ae,
                np.array(train_labels), 
                X_target_feats,
                X_target_ae,
                ae_input_inds,
                ae_output_inds,
                X_unlabeled_feats=X_unlabeled_feats,
                X_unlabeled_ae=X_unlabeled_ae,
                y_target=None,  # If we come up with oracle mode this can be non-None
                X_source_valid_feats=X_source_valid_feats,
                X_source_valid_ae=X_source_valid_ae,
                y_source_valid=np.array(test_labels))

    best_model = torch.load('best_model.pt')

    device='cuda' if torch.cuda.is_available() else 'cpu'

    target_X_feats = torch.FloatTensor(X_target_feats.toarray()).to(device)
    target_X_ae = torch.FloatTensor(X_target_ae[:,ae_input_inds].toarray()).to(device)
    target_y = np.array(dest_test_labels)
    if args.eval == 'pt':
        target_test_predict_raw,_,_ = best_model(target_X_feats, target_X_ae)
        target_test_predict = np.round(sigmoid(target_test_predict_raw).data.cpu().numpy())[:,0]
        correct_preds = (target_y == target_test_predict).sum()
        acc = correct_preds / len(target_y)
    # elif args.eval == 'lr':
        # c_parm = 0.1
        # logreg =  LogisticRegression(C=c_parm)
        # source_X_pt = torch.FloatTensor(X_source_feats).to(device)
        # source_X_np_pt = torch.FloatTensor(X_source_ae[:, non_pivot_candidate_inds])
        # source_reps = best_model(source_X_pt, source_X_np_pt)[2].data.cpu().numpy()
        # allfeatures = np.concatenate( (X_train_source_un_encoded, source_reps), axis=1)
        # # Train with source features:
        # logreg.fit(allfeatures, train_labels)

        # target_reps = best_model(target_X, target_X_nonpivot)[2].data.cpu().numpy()
        # allfeaturesFinal = np.concatenate( (X_2_test_target_un_encoded, target_reps), axis=1)
        # acc = logreg.score(allfeaturesFinal, dest_test_labels)


    # tps = (target_test_predict * target_y).sum().item()
    # test_true_labels = target_y.sum()
    # test_true_preds = target_test_predict.sum()
    # rec = tps / test_true_labels
    # prec = tps / test_true_preds
    # f1 = 2 * rec * prec / (rec + prec)
    log("Target accuracy=%f" % (acc))



if __name__ == '__main__':
    main(sys.argv[1:])

