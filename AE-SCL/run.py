import tr
import sentiment
from shutil import copyfile
import gc
from keras import backend as K
import argparse

parser = argparse.ArgumentParser(description="AE-SCL starting script")
parser.add_argument('--iters', type=int, default=1, help='Number of iterations to run for')
parser.add_argument('--pivots', type=int, default=100, help='Number of pivot features to use (max)')
parser.add_argument('--nhid', type=int, default=500, help='Number of hidden nodes in neural network that learns pivot features')
parser.add_argument('--method', type=str, default='mi', help='Method to use for pivot selection of: {mi, random, list, learned, oracle}')
parser.add_argument('--pivot-file', type=str, default='', help='File to use for reading pivot features')
parser.add_argument('--doc-freq', type=int, default=10, help='Minimum required document frequency for pivot features')

def read_pivot_file(fn):
    pivots = []
    with(open(fn, 'r')) as f:
        for line in f.readlines():
            vals = line.rstrip().split(' ')
            if len(vals) < 3:
                continue
            name = vals[2]
            feat = [str(x) for x in name.split('_')]
            
            if feat[0] == 'Bigram':
                pivots.append(' '.join(feat[1:]))
            elif feat[0] == 'Unigram':
                pivots.append(feat[1])

    return pivots

if __name__ == '__main__':
    args = parser.parse_args()
    for ind in range(args.iters):
        domain = [] 
        domain.append("books")
        domain.append("dvd")
        domain.append("electronics")
        domain.append("kitchen")

        source_ind = 3
        target_ind = 1

        # making a shared representation for both source domain and target domain
        # first param: the source domain
        # second param: the target domain
        # third param: number of pivots
        # fourth param: appearance threshold for pivots in source and target domain
        # fifth parameter: the embedding dimension, identical to the hidden layer dimension

        pivots = []
        if args.method == 'list':
            pivots = read_pivot_file(args.pivot_file)
            
        tr.train(domain[source_ind], domain[target_ind], args.pivots, args.doc_freq, args.nhid, pivot_method=args.method, pivots=pivots)

        # learning the classifier in the source domain and testing in the target domain
        # the results, weights and all the meta-data will appear in source-target directory
        # first param: the source domain
        # second param: the target domain
        # third param: number of pivots
        # fourth param: appearance threshold for pivots in source and target domain
        # fifth param: the embedding dimension identical to the hidden layer dimension
        # sixth param: we use logistic regression as our classifier, it takes the const C for its learning

        sentiment.sent(domain[source_ind], domain[target_ind], 100, 10, 500, 0.1)
        print("Running AE-SCL with source domain %s and target domain %s" % (domain[source_ind], domain[target_ind]))
        
        copyfile("%s_to_%s/pivot_names/pivot_names_%s_%s_100_10" % (domain[source_ind], domain[target_ind], domain[source_ind], domain[target_ind]),
                 "%s_to_%s/pivot_names/pivot_names_%s_%s-100_10_%d" % (domain[source_ind], domain[target_ind], domain[source_ind], domain[target_ind], ind))
        gc.collect()
        K.clear_session()


