1. I use python 3.5
2. I use the pandas, numpy, scipy libray. So you need install these libraries first.
3. I use the sparse matrix csr_matrix to do the matrix computation.
4. stemming is slow. So I did some preprocess.
5. the CorpusReader_TFIDF() class the __init__ maybe we are not very same. To clarify, 
tf = "raw" or "log" or "binary" choose tf computation function.
idf= "inverse" or "smoothed" or "probabilistic" choosse idf computation function.
You should use dict={'sopword':"no" ......} to create the object.
6.TFIDF are the files of the class CorpusReader_TFIDF.
7. If you just run these 3 corpuses, it just needs about 6 seconds. I speed on stemming and matrix.

8. for my program it will have such warning for corpus 2. but I alreday use np.where to deal with the log 0 case. I search the web, it seems it already handled that case with the warning. 
So it is ok to see the warning.
corpus2:
/home/free/git/nlprog2/TFIDF.py:271: RuntimeWarning: divide by zero encountered in log2
  ret= np.where(ret!=0, np.log2(ret) + 1, 0)
9. My program will run about 15 minutes.