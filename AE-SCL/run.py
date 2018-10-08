import tr
import sentiment
if __name__ == '__main__':
    domain = []
    domain.append("books")
    domain.append("dvd")
    domain.append("electronics")
    domain.append("kitchen")

    source_ind = 2
    target_ind = 3

    # making a shared representation for both source domain and target domain
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth parameter: the embedding dimension, identical to the hidden layer dimension

    tr.train(domain[source_ind], domain[target_ind], 100, 10, 500)

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

