import numpy as np
import scipy as sp
import nltk
from nltk.stem import WordNetLemmatizer
import re
import unicodedata

# input : sentence
# 1) For every two words, 
# find difference(for GloVe) wrt each other
# 2) Decide whether each pair is antonym/synonym
# by comparing L2distance between ready-made
# antonym/synonym vectors
# 3) For chosen pairs, check dependency
# 4) If dependent,check negation around each of the pair
# Note that full count can have similar result
# 5) Decision by reflecting negation

cont   = open('contradiction.txt').read().splitlines()
contf  = open('contradiction-free.txt').read().splitlines()
antset = open('antonym.txt').read().splitlines()
numant = len(antset)
synset = open('synonym.txt').read().splitlines()
numsyn = len(synset)
NEG    = ['no', 'not', 'never', "n't"]
Postag = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def loadvector(File):
    print "Loading word vectors"
    f = open(File,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

#lexvec300 = loadvector('lexvec300.txt')
#glove100  = loadvector('glove100.txt')
#glove25   = loadvector('glove25.txt')
glove50    = loadvector('glove50_wiki.txt')

wordsize  = 50
dict = glove50

def make_antvec():
   antvec = np.zeros((numant,wordsize))
   for i in range(numant):
     x = antset[i].lower().split()
     if x[0] in dict and x[1] in dict:
      y = dict[x[0]]-dict[x[1]]
      antvec[i,:] = y
      print i
   #np.save(antvec,antvec_lex300)
   return antvec

antvec = make_antvec()

def make_synvec():
   synvec = np.zeros((numsyn,wordsize))
   for i in range(numsyn):
     x = synset[i].lower().split()
     if x[0] in dict and x[1] in dict: 
      y = dict[x[0]]-dict[x[1]]
      synvec[i,:] = y
      print i
   #np.save(synvec,synvec_lex300)
   return synvec

synvec = make_synvec()

#antvec = np.load('antvec_lex300.npy')
#synvec = np.load('synvec_lex300.npy')

Dant=0.41
Dsyn=0.43
lmt = WordNetLemmatizer()

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# decides if input sentence contains contradiction or not
def cont_stat_decision(s):
   x = tokenizer.tokenize(s)
   text = nltk.Text(x)
   tags = nltk.pos_tag(text)
   L = len(x)
# tag pos and lemmatize
   tagproj = []
   for k in range(L):
      tagproj.append(tagger(tags[k][1]))
      if tagproj[k] != 'pass':
         x[k] = str(lmt.lemmatize(x[k],tagproj[k]))
# embed word vectors
   X = np.zeros((L,wordsize))
   for k in range(L):
      if x[k] in dict:
          X[k,:] = dict[x[k]]
          #print x[k]
   print "Length of sentence:{}".format(L)
# ind = 0   
   dec = 0
   for i in range(L-1):
      if dec ==0:
         for j in range(i+1,L):
            ii = 0
            jj = 0
            for pt in Postag:
               if pt == tags[i][1]:
                  ii = 1
               if pt == tags[j][1]:
                  jj = 1
            if ii*jj == 1:
               #print(x[i],x[j])
               if x[i] != 'be' and x[j] != 'be':
#                 dep = check_dependency(s,i,j)
                  dec = cont_decision(s,X[i],X[j],negation(x,i),negation(x,j))
   if dec==1:
      print("Contradiction!")
   return dec

def tagger(f):
   if f in ['NN', 'NNS']:
      return 'n'
   elif f in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
      return 'v'
   else:
      return 'pass'

# decides if the pair are ant/sym
# regarding 3 functions below
def cont_decision(s,x,y,negx,negy):
   if antsyn_check(x,y)>=0:
      temp = antsyn_check(x,y)+negx+negy
      return np.remainder(temp,2)
   else: return 0

csd = sp.spatial.distance.cosine

def mesant(x,y):
   z = x - y
   score = 1.5
   for row in antvec:
      d = min(csd(row,z),csd(row,-z))
      if d<score:
         score = d
   return score

def messyn(x,y):
   z = x-y
   score = 1.5
   for row in synvec:
      d = min(csd(row,z),csd(row,-z))
      if d<score:
         score = d
   return score

# checks if the pair are ant/sym
def antsyn_check(x,y):
   if mesant(x,y)<Dant:
      return 1
   elif messyn(x,y)<Dsyn:
      return 0
   else: return -1

# checks if the pair are dependent
def check_dependency(s,i,j):
   x = nltk.word_tokenize(s)
#  if dependent
#     return 0
#  else return 1

# negation count for no, not, never, nothing        
def check_negation(w):
   negcount = 0
   for n in NEG:
      if w.lower() == n:
         negcount = 1
         return 1
         break
   if negcount==0:
      return 0

# returns 0 if negation comes 1 or 2 word before
def negation(x,k):
   if k==0:
      return 0
   if k==1:
      return check_negation(x[k-1])
   if k>=2:
      return np.remainder((check_negation(x[k-2])+check_negation(x[k-1])),2)

def test_oxy():
 total=0
 corr =0
 for i in range(len(cont)):
   st = cont[i].replace("\xa1\xaf","'")
   st = st.replace("\xa1\xae","'")
   st = st.lower()
   print (i,st)
   corr  = corr + cont_stat_decision(st)
   total = total + 1
   print (corr,total)
   accuracy = corr/float(total)
   print accuracy,"\n"
 return corr, float(total)

def test_noxy():
 totalf=0
 corrf =0
 for i in range(len(contf)):
   st = contf[i].replace("\xa1\xaf","'")
   st = st.replace("\xa1\xae","'")
   st = st.lower()
   print (i,st)
   corrf  = corrf + (1-cont_stat_decision(st))
   totalf = totalf + 1
   print (corrf,totalf)
   accuracy = corrf/float(totalf)
   print accuracy,"\n"
 return corrf, float(totalf)

corr, total  = test_oxy()
corrf, totalf = test_noxy()

pre = corr/(corr+(totalf-corrf))
print 'Precision:', pre
rec = corr/total
print 'Recall:', rec
print 'F-measure:', 2*pre*rec/(pre+rec)
print 'Accuracy:', (corr+corrf)/(total+totalf)

