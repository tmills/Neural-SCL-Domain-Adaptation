# Neural Structural Correspondence Learning for Domain Adaptation.
Authors: Yftah Ziser, Roi Reichart (Technion - Israel Institute of Technology).

This is a code repository used to generate the results appearing in [Neural Structural Correspondence Learning for Domain Adaptation](https://www.aclweb.org/anthology/K/K17/K17-1040.pdf).

If you use this implementation in your article, please cite :)
```bib
@InProceedings{ziser-reichart:2017:CoNLL,
  author    = {Ziser, Yftah  and  Reichart, Roi},
  title     = {Neural Structural Correspondence Learning for Domain Adaptation},
  booktitle = {Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)},
  year      = {2017},  
  pages     = {400--410},	
}
```

You can find detailed instructions for using the AE-SCL and AE-SCL-SR models(including Prerequisites) in their corresponding directories.

You can find an explained example in model_name\run.py,
e.g., AE-SCL-SR\run.py : 

```python
import tr
import sentiment
if __name__ == '__main__':
    domain = []
    domain.append("books")
    domain.append("kitchen")
    domain.append("dvd")
    domain.append("electronics")

    # making a shared representation for both source domain and target domain
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth parameter: the embedding dimension, identical to the hidden layer dimension

    tr.train(domain[0], domain[1], 100, 10, 500)

    # learning the classifier in the source domain and testing in the target domain
    # the results, weights and all the meta-data will appear in source-target directory
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth param: the embedding dimension identical to the hidden layer dimension
    # sixth param: we use logistic regression as our classifier, it takes the const C for its learning

    sentiment.sent(domain[0], domain[1], 100, 10, 500, 0.1)


```
