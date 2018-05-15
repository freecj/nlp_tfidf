import nltk
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import xlogy
from scipy.sparse import diags
from nltk.corpus import stopwords
from collections import Counter
from numpy.linalg import norm
import math
class CorpusReader_TFIDF():
    """
    construct a sparse matrix for tf ,and another binary csr_matrix for idf computation

    """

    def transform(self):

        # result of document conversion to term count dicts
        term_counts_per_doc = []

        term_counts = Counter()
        for doc in self.texts:
            term_count_current = Counter(doc)
            term_counts.update(term_count_current)
            term_counts_per_doc.append(term_count_current)
            # print('\n')
            # print (term_count_current)
        terms = set(term_counts)
        sortedterms = sorted(terms)
        vocab = dict(((t, i) for i, t in enumerate(sortedterms)))
        if not vocab:
            raise ValueError("empty vocabulary; training set may have"
                             " contained only stop words.")

        # print (vocab)
        self.termname = sortedterms
        # print()
        indptr = [0]
        indices = []
        data = []
        vocabulary = vocab
        self.vocab = vocab
        for i, term_count_dict in enumerate(term_counts_per_doc):
            for term, count in term_count_dict.items():
                j = vocabulary.get(term)
                indices.append(j)
                data.append(count)
                # print(count)
            # free memory as we go
            indptr.append(len(indices))
            term_count_dict.clear()

        self.tfmatrix = csr_matrix(
            (data, indices, indptr), shape=(len(self.texts), len(vocabulary)),
            dtype=float)
        # binary matrix for computing idf
        self.binarymatrix = self.tfmatrix.copy()
        self.binarymatrix.data.fill(1)

        # self.print_matrix(self.binarymatrix)

    """
    term frequency: raw
    """

    def raw(self):

        self.texts = tuple((self.corpus.words(file))
                           for file in self.filename)

        # print(self.texts)
        np.set_printoptions(precision=5, suppress=False)
        np.set_printoptions(threshold=np.inf)
        self.texts = self.stemmingTokenizer()

        self.transform()
        # self.print_matrix(self.tfmatrix)
        #self.vocabulary = vocabulary
        # np.savetxt("result.txt",)
        # print(self.tfmatrix)
        # print(self.tfmatrix)

    """
    term frequency: log normalized
    """

    def logfreq(self):
        self.texts = [(self.corpus.words(file))
                      for file in self.filename]

        # print(self.texts)
        np.set_printoptions(precision=5, suppress=False)
        np.set_printoptions(threshold=np.inf)
        self.texts = self.stemmingTokenizer()
        self.transform()

        self.tfmatrix.data = np.log2(self.tfmatrix.data)
        self.tfmatrix.data += 1
        # self.print_matrix(self.tfmatrix)

    """
    binary term frequency
    """

    def binary(self):
        self.texts = [set(self.corpus.words(file))
                      for file in self.filename]

        # print(self.texts)
        np.set_printoptions(precision=5, suppress=False)
        np.set_printoptions(threshold=np.inf)
        self.texts = self.stemmingTokenizer()
        # print(self.texts)
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}
        termname = []
        for d in self.texts:
            for term in d:
                index = vocabulary.setdefault(term, len(vocabulary))
                if (index == len(vocabulary) - 1):
                    termname.append(term)
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
        self.tfmatrix = csr_matrix(
            (data, indices, indptr), shape=(len(self.texts), len(vocabulary)),
            dtype=np.float16)
        # print(termname)
        self.termname = termname
        self.binarymatrix = self.tfmatrix
        # self.vocabulary = vocabulary
        # np.savetxt("result.txt",)
        # print(self.tfmatrix)
        # self.print_matrix(self.tfmatrix)
        # print(self.tfmatrix)
    """
    print csr_matrix (sparse) less than tf_idf_dim()
    """

    def print_matrix(self, matrix):
        printmatrix = matrix.copy()
        bool_vect = []
        lengthdim = len(self.tf_idf_dim())
        for i in range(len(self.termname)):
            if i < lengthdim:
                bool_vect.append(1)
            else:
                bool_vect.append(0)
        indices = np.where(bool_vect)[0]

        out1 = printmatrix.tocsc()[:, indices]

        results = pd.DataFrame(
            out1.toarray(), index=self.filename, columns=self.tf_idf_dim())
        print(results)
    '''
    inverse document frequency: inverse
    '''

    def inverse(self):

        idfmatrix = (len(self.texts) / self.binarymatrix.sum(axis=0))
        idfmatrix.astype(np.float16)

        idfmatrix = xlogy(np.sign(idfmatrix), idfmatrix) / np.log(2)
        # print(idfmatrix)

        data = np.asarray(idfmatrix).reshape(-1)

        self.idfmatrix = diags(data, 0)

        self.tfidfmatrix = self.tfmatrix * self.idfmatrix
        # self.print_matrix(self.tfidfmatrix)
    '''
    inverse document frequency: smoothed (inverse smoothed)
    '''

    def smoothed(self):
        idfmatrix = (len(self.texts) / self.binarymatrix.sum(axis=0)) + 1
        idfmatrix.astype(np.float16)

        idfmatrix = xlogy(np.sign(idfmatrix), idfmatrix) / np.log(2)
        # print(idfmatrix)

        data = np.asarray(idfmatrix).reshape(-1)

        self.idfmatrix = diags(data, 0)

        self.tfidfmatrix = self.tfmatrix * self.idfmatrix
        # self.print_matrix(self.tfidfmatrix)
    """
    inverse document frequency: probabilistic (pro abilistic inverse)
    """

    def probabilistic(self):
        # use binary tf matrix to compute the idf ni (sum function of csr_matrix)
        idfmatrix = (len(self.texts) - self.binarymatrix.sum(axis=0)
                     ) / self.binarymatrix.sum(axis=0)
        # print(idfmatrix)
        idfmatrix.astype(np.float16)

        idfmatrix = xlogy(np.sign(idfmatrix), idfmatrix) / np.log(2)
        # print(idfmatrix)

        data = np.asarray(idfmatrix).reshape(-1)

        self.idfmatrix = diags(data, 0)

        self.tfidfmatrix = self.tfmatrix * self.idfmatrix
        # self.print_matrix(self.tfidfmatrix)

    '''
    -tf_idf(): return a list of ALL tf-idf vector (each vector should be a list) for the corpus,ordered by the order where fields are returned (the dimensions of the vector can be
    arbitrary, but need to be consistent with all vectors)

    - tf_idf(fileid=fileid): return the tf-idf vector corresponding to that file

    - tf_idf(filelist=[fileid]): return a list of vectors, corresponding to the tf-idf to the list of
    fileid input
    '''

    def tf_idf(self, fileid=None):
        if fileid is None:
            temp = self.tfidfmatrix.copy()
            return temp.todense().tolist()
        else:
            ret = []
            if type(fileid) is not list:
                fileid = [fileid]
            for item in fileid:
                if item in self.filename:
                    ret.append((self.tfidfmatrix.getrow(self.filename.index(
                        item)).toarray().tolist()[0]))

            return ret

    """
    return the list of the words corresponding to each vector of the tf-idf
    vector
    """

    def tf_idf_dim(self):
        return self.termname[0:15]
    """
    the input should be a list of words (treated as a document). The
    function should return a vector corresponding to the tf_idf vector for the new
    document
    """

    def tf_idf_new(self, words):
        newwords = self.stemmingTokenizer(words)[0]
        self.newwords = newwords
        ret = np.array([0 for i in range(len(self.termname))], dtype=float)
        # print(newwords)
        if self.tf == "binary":
            newwords = set(newwords)
            for word in newwords:
                j = self.vocab.get(word)
                if j is not None:
                    ret[j] = 1
        else:
            for word in newwords:
                j = self.vocab.get(word)
                if j is not None:
                    ret[j] += 1
        # print(ret)
        if self.tf == "log":
            ret= np.where(ret!=0, np.log2(ret) + 1, 0)
        # print(ret)
    
        return (ret * self.idfmatrix).tolist()

    def cosine_sim(self, fileid):

        if len(fileid) < 2:
            return None
        else:
            if fileid[0] not in self.filename or fileid[1] not in self.filename:
                return None
            else:
                if self.filename.index(fileid[0]) == self.filename.index(fileid[1]):
                    return 1
                vectorf1 = self.tfidfmatrix.getrow(
                    self.filename.index(fileid[0]))
                vectorf2 = self.tfidfmatrix.getrow(
                    self.filename.index(fileid[1]))
                vectorf1 = vectorf1.toarray()[0]
                vectorf2 = vectorf2.toarray()[0]
                # print(vectorf1)
                # print(vectorf2)
                valuef1 = vectorf1
                valuef2 = vectorf2
                valuef1 = np.sqrt(valuef1.T.dot(valuef1))
                valuef2 = np.sqrt(valuef2.T.dot(valuef2))
                # print(valuef1)
                # print(valuef2)
                cossim = vectorf1.T.dot(vectorf2) / valuef1
                cossim = cossim / valuef2
                return cossim

    def fileids(self):
        return self.corpus.fileids()

    """
    cosine_sim_new([words], fileid): the [words] is a list of words as is in the parameter of
    tf_idf_new() method. The fileid is the document in the corpus.
    """

    
    def cosine_sim_new(self, words, fileid):
        vectorf1 = np.asarray(self.tf_idf_new(words))
        vectorf2 = self.tfidfmatrix.getrow(
            self.filename.index(fileid))
        vectorf2 = vectorf2.toarray()[0]
        normvaluea = norm(vectorf1)
        if normvaluea == 0:
            return 0
        cossim = np.dot(vectorf1, vectorf2)/(normvaluea*norm(vectorf2))
        # print(vectorf1)
        # print(vectorf2)
        '''
        valuef1 = vectorf1
        valuef2 = vectorf2
        
        valuef1 = np.sqrt(valuef1.T.dot(valuef1))
        if valuef1 == 0:
            return 0
        valuef2 = np.sqrt(valuef2.T.dot(valuef2))
        # print(valuef1)
        # print(valuef2)
        cossim = vectorf1.T.dot(vectorf2) / valuef1
        cossim = cossim / valuef2
        '''
        return cossim
    """
    stopword (keyword): if specified as “none”, then do not remove any stopwords.
    Otherwise this should treat as a filename where stopwords are to be read. Default is
    using the standard English stopwords corpus in NLTK. You should assume any word
    inside the stopwords file is a stopword. Otherwise you should not assume any predefined
    format of the stopword file.

    """

    def getStopFile(self, stopword):
        words = []
        if stopword == "none":
            return words
        with open(stopword, 'r') as f:
            line = f.read()
        words = line.split()
        return words

    """
    stemmer (keyword): the stemming function to be used. Default is to use the Porter
    stemmer (nltk.stem.porter)
    """

    def stemmingTokenizer(self, newfile=None):
        ret = []
        
        termname = []
        dict = set()
        stemdict = {}
        stemset = set()
        if newfile is None:
            texts = self.texts
        else:
            texts = [newfile]
        self.stopword = frozenset(self.stopword)
        for doc in texts:
            if self.ignorecase == "yes":
                words = [
                    item.lower() for item in doc if item not in self.stopword and item.isalnum()]
                #words = [item.lower() for item in doc if item.isalnum()]
                #words = [self.stemmer.stem(word) for word in words]
                wo = []
                if self.stemmer == "none":
                    if self.binaryfreq == 1:
                        ret.append(set(words))
                    else:
                        ret.append(words)
                    continue
                for stemword in words:
                    if stemword in dict:
                        wo.append(stemword)
                    else:

                        flag = stemdict.get(stemword)
                        if flag is None:

                            temp = self.stemmer.stem(stemword)
                            dict.add(temp)
                            # stemset.add(stemword)
                            stemdict[stemword] = temp
                            wo.append(temp)
                        else:
                            wo.append(stemdict[stemword])
                # print(wo)
                #words = list(map(self.stemmer.stem, words))
            else:
                words = [word for word in doc
                         if word not in self.stopword and word.isalnum()]
                if self.stemmer == "none":
                    if self.binaryfreq == 1:
                        ret.append(set(words))
                    else:
                        ret.append(words)
                    continue
                wo = []
                for stemword in words:
                    if stemword in dict:
                        wo.append(stemword)
                    else:
                        flag = stemdict.get(stemword)
                        if flag is None:

                            temp = self.stemmer.stem(stemword)
                            dict.add(temp)
                            # stemset.add(stemword)
                            stemdict[stemword] = temp
                            wo.append(temp)
                        else:
                            wo.append(stemdict[stemword])
            
   
            # if just binary frequecy, let list be set to speed
            if self.binaryfreq == 1:
                ret.append(set(wo))
            else:
                ret.append(wo)

        return ret

    """
    Constructor: Your constructor should have the following parameters
    - corpus (required): should be a corpus object

    - tf (keyword): the method used to calculate term frequency. Default is frequency (use
    the name specified above)

    - idf (keyword): the method used to calculate inverse document frequency. Default is
    Base (use the name specified above)

    - stopword (keyword): if specified as “none”, then do not remove any stopwords.
    Otherwise this should treat as a filename where stopwords are to be read. Default is
    using the standard English stopwords corpus in NLTK. You should assume any word
    inside the stopwords file is a stopword. Otherwise you should not assume any predefined
    format of the stopword file.

    - stemmer (keyword): the stemming function to be used. Default is to use the Porter
    stemmer (nltk.stem.porter)

    - ignorecase (keyword) if specified as “no”, then do NOT ignore case. Default is ignore
    case
    """

    def __init__(self, corpus, stopword=None, tf="raw", idf="inverse",
                 stemmer=None, ignorecase="yes"):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        self.tf = tf
        self.idf = idf
        self.ignorecase = ignorecase
        if stopword is None:
            self.stopword = set(stopwords.words('english'))
        else:
            self.stopword = set(self.getStopFile(stopword))

        if stemmer is None:
            self.stemmer = nltk.stem.PorterStemmer()
        else:
            self.stemmer = stemmer
        self.corpus = corpus
        self.filename = corpus.fileids()
        if tf == "binary":
            self.binaryfreq = 1
            self.binary()
        elif tf == "log":
            self.binaryfreq = 0
            self.logfreq()
        else:
            self.binaryfreq = 0
            # print("raw")
            self.raw()

        if idf == "smoothed":
            self.smoothed()
        elif idf == "probabilistic":
            self.probabilistic()
        else:
            # print("inverse")
            self.inverse()
