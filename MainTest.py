import sys
import os
sys.path.append(os.getcwd())
import nltk.corpus as corpus
from nltk.corpus import brown
from nltk.corpus import shakespeare  # XMLCorpusreader
from nltk.corpus import state_union

from TFIDF import CorpusReader_TFIDF as tfidftool
from nltk.corpus import PlaintextCorpusReader

import time as time
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import heapq
import operator


'''

Corpus is a CorpusReader_TFIDF object
 q is the string that forms the query
 rank is a keyword parameter, if present, the function should return the top <rank> documents
    from the query. If rank is negative, then the output should be ranked. (e.g. -15 will returned the
    top 15 documents ranked by cosine similarity [from most similar to least similar]
 minsim is a keyword parameter, if present, the function should return all documents that the
    have cosine similarity of at least minsim
    (If both rank and minsim are presence, than both condition are to be satisfied)
    (If neither parameters are mentioned, the function do nothing and return an empty list)

    The function is to return the list of documents (based on the index in the Corpus).
'''
def Retrieve(corpus, q, rank = None, minsim = None):
    ret = []
    if (rank is None and minsim is None):
        return ret
    wordlist = q.split(None)
    #print(wordlist)
    h = []
    Dict = {}
    if (minsim is None and rank is None):
        return ret
    else:
        for fileid in corpus.fileids():

            coss = corpus.cosine_sim_new(wordlist,fileid)
            if minsim is not None:
                if coss < float(minsim):
                    continue
            #h.append((corpus.cosine_sim_new(wordlist,fileid), fileid))i
            '''
            if rank is not None:
                capacity = abs(int(rank))
                if len(h) < capacity:
                    heapq.heappush(h,(coss, fileid))
                else:
                    heapq.heappushpop(h, (coss,fileid))
            else:
            '''
            heapq.heappush(h,(coss, fileid))
            #Dict[fileid] = corpus.cosine_sim_new(wordlist,fileid)sorted
        #after = sorted(h, key=operator.itemgetter(0), reverse=True)
    #expensive = heapq.nlargest(3, h, key=lambda s: s[0])
    #print(rank)
    #print(after)
    if (rank is not None) :
        capacity = abs(int(rank))
        h = heapq.nlargest(capacity, h, key = lambda s: s[0])
    '''
    if rank is not None:
        required_cnt = abs(int(rank))
        expensive = h#heapq.nlargest(required_cnt, h, key=lambda s: s[0])
    else:
        ret = [i[1] for i in h]
        return ret
    cnt = 0
    print(expensive)
    for value in after:
        cnt += 1
        if cnt > required_cnt:
            break
        ret.append(value[1])
        #print("{}:{}".format(value[1], value[0]))
    '''
    ret = [i[1] for i in h]
    #print(ret)

    return ret
# print the information


def readquery(filename):
    querylist = []
    dict={}
    with open(filename) as fileConfig:
        for line in fileConfig:
            temp = line.split(None, 1)
            dict[temp[0]] = temp[1]
    fileConfig.close
    return dict


def readresult(filename):
    querydict = defaultdict(list)
    querylist = []
    with open(filename) as fileConfig:
        for line in fileConfig:
            temp = line.split(None, 1)
            #print(temp[1])
            querydict[temp[0]].append(temp[1].split('\n')[0])

    fileConfig.close
    return querydict


def test():
    strtest = []
    fileaddre = os.path.abspath(os.path.dirname(__file__)) + '/'
    # open configure file
    with open(fileaddre + 'configure.txt') as fileConfig:
        for counter, line in enumerate(fileConfig):
            if (counter < 3):
                strtest.append(fileaddre + line.split(None)[0])
            elif (counter < 5):
                #print(line)
                strtest.append(line.split(None)[:2])
    if len(strtest) <= 3:
        print("just has <=3 lines, error!")
        return
    #print(counter)
    #print (strtest)
    fileConfig.close
    #print(strtest)
    dirlist = []
    #open query file
    for root, dirs, files in os.walk(strtest[0]):
        for filename in files:
            dirlist.append(filename)
    #print (len(dirlist))
    querydict = readquery(strtest[1])
    querynamelist = []
    keyname = sorted(querydict.copy().keys())

    #print(keyname)
    for k in keyname:
        name = 'query'+k
        #print(name)
        querynamelist.append(name)
    #print(querydict)
    #open result file
    resultdict = readresult(strtest[2])
    #print(resultdict)
    #print (strtest[3][1])
    paramdict = [
        {'tf': "raw", 'idf': "inverse", 'stopword': "none", 'stemmer': "none"},
        {'tf': "log", 'idf': "smoothed", 'stopword': "none", 'stemmer': "none"},
        {'tf': "raw", 'idf': "inverse"},
        {'tf': "raw", 'idf': "smoothed"},
        {'tf': "log", 'idf': "inverse"},
        {'tf': "log", 'idf': "smoothed"}
            ]
    corpus_root = strtest[0]
    wordlists = PlaintextCorpusReader(corpus_root, '.*')
    xre = []
    ypre = []
    cnt = 1;
    groupname = []
    
    for paramitem in paramdict:
        corpustest = tfidftool(wordlists,**paramitem)

        print("corpus%s:" % (cnt))
        stra = 'corpus' + str(cnt)
        #print (stra)
        groupname.append(stra)
        myretdict = defaultdict(list)
        for k, v in querydict.items():
            
            ransim = strtest[3][1]
            if strtest[3][0] == 'r':
                #print(strtest[3][0])
                paramretrieve = {'rank': ransim}
            elif strtest[3][0] == 's':
                #print(strtest[3][0])
                paramretrieve = {'minsim': ransim}
            else:
                paramretrieve = {}
            myretdict[k].extend(Retrieve(corpustest , v, **paramretrieve))
          
        #for key in sorted(myretdict.keys()):
        #    print ("%s: %s" % (key, myretdict[key]))
        recalldict = {}
        precidict = {}
        recalllist = []
        precilist = []
        recallde = 0
        recallno = 0
        preci = 0
        #print(resultdict)
        for k, v in resultdict.items():
            a = set (myretdict[k])
            #print(k)
            #print(myretdict[k])
            b = set (v)
            #print(b)
            #print(a)
            #print(b.intersection(a))
            precidict[k] = len(b.intersection(a)) / len(a) if len(a) != 0 else 0

            recalldict[k] = len(b.intersection(a)) / len(b) if len(b) !=0 else 0
            recallde += len(b)
            recallno += len(b.intersection(a))
            preci += len(a)
        #print(recallno/recallde)
        #print(recallno/preci)
        for k in keyname:
            if (recalldict.get(k) is not None):
                recalllist.append(recalldict[k])
            else :
                recalllist.append(0)
            if precidict.get(k) is not None:
                precilist.append(precidict[k])
            else:
                precilist.append(0)

        averpre = 0
        averrecall = 0
        for k in precilist:
            averpre += k
        for k in recalllist:
            averrecall += k
        averpre /= len(precilist)
        averrecall /= len(recalllist)
        namecopy = querynamelist[:]
        namecopy.append("average") 
        namecopy.append("overall")
        recalllist.append(averrecall)
        precilist.append(averpre)
        r = (recallno/recallde) if recallde != 0 else 0

        p = (recallno/preci) if preci != 0 else 0
        recalllist.append(r)  
        precilist.append(p)
        xre.append(r)
        ypre.append(p)
        out = pd.DataFrame({'Query':namecopy, 'recall':recalllist, 'precision':precilist})
        print(out)
        cnt  +=1;

if __name__ == "__main__":
    test()
