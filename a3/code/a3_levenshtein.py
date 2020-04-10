import os
import numpy as np
import sys
import re

dataDir = '/u/cs401/A3/data/'
dataDir = "/scratch/ssd001/home/cchoquet/csc401/a3/code/data/data/"


def add_tags(str_list):
    str_list.insert(0, '<s>')
    str_list.insert(len(str_list), '</s>')
    return str_list


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    # create and initialize our grids
    # dim 0: for r
    # dim 1: for h
    # dim 2: 2D where 0 index is the distance and 1 index is the type
    # types are 0: delete, 1: insert, 2: substitute
    nS = 0
    nI = 0
    nD = 0
    n = len(r)
    m = len(h)
    r.insert(0, '<s>')
    r.append('</s>')
    h.insert(0, '<s>')
    h.append('</s>')
    R = np.zeros([n + 2, m + 2])
    R[0] = np.arange(m + 2)
    R[:, 0] = np.arange(n + 2)
    for i in range(1, n + 2):
        for j in range(1, m + 2):
            substitution = 1
            if r[i] == h[j]:
                substitution = 0
            R[i, j] = np.min([R[i - 1, j] + 1, R[i - 1, j - 1] + substitution, R[i, j - 1] + 1])

    i = n + 1
    j = m + 1
    while i != 0 or j != 0:
        if R[i - 1, j - 1] == R[i, j] and i > 0 and j > 0:
            i, j = i - 1, j - 1
        elif R[i - 1, j - 1] + 1 == R[i, j] and i > 0 and j > 0:
            nS += 1
            i, j = i - 1, j - 1
        elif R[i, j - 1] + 1 == R[i, j] and j > 0:
            nI += 1
            j = j - 1
        elif R[i - 1, j] + 1 == R[i, j] and i > 0:
            nD += 1
            i = i - 1
        # else:
        #     raise AssertionError("Not any of the four cases.")

    return R[-1, -1] / n, nS, nI, nD


def process_line(l):
    return re.sub(r"[^a-z\[\] ]+", "", l.lower()).split()


def make_print(speaker, t, i, out):
    beg = [speaker, t, i, out[0]]
    prefixes = ['S:{}', 'I:{}', 'D:{}']
    return beg + list([p.format(o) for p, o in zip(prefixes, out[1:])])


if __name__ == "__main__":
    sys.stdout = open('asrDiscussion.txt', 'w')
    g_lev = []  # google levenshtein data
    k_lev = []  # kaldi levenshtein data
    transcripts = ['transcripts.txt', 'transcripts.Google.txt', 'transcripts.Kaldi.txt']
    for root, ds, fs in os.walk(dataDir):
        for speaker in ds:
            print(speaker)
            paths = [os.path.join(root, speaker, t) for t in transcripts]
            try:
                alllines = zip(*[open(p, 'r').readlines() for p in paths])
            except:
                continue
            for i, opened_transcripts in enumerate(alllines):
                if any([len(x) == 0 for x in opened_transcripts]):
                    continue
                r, g, k = opened_transcripts
                r = process_line(r)
                print(r)
                print(process_line(g))
                print(process_line(k))
                goog = Levenshtein(r, process_line(g))
                g_lev.append(goog)
                goog = make_print(speaker, 'Google', i, goog)
                kald = Levenshtein(r, process_line(k))
                k_lev.append(kald)
                kald = make_print(speaker, 'Kaldi', i, kald)
                s = " ".join(["{}" for _ in range(7)])
                print(s.format(*goog))
                print(s.format(*kald))
                break
            break
        break
    g_lev = np.array(g_lev)
    k_lev = np.array(k_lev)
    out1 = f"Google has a mean of: {np.mean(g_lev[:, 0])} " \
        f"and std_dev of: {np.sqrt(np.var(g_lev[:, 0]))}. Kaldi has a mean of: " \
        f"{np.mean(k_lev[:, 0])} and std_dev of: " \
        f"{np.sqrt(np.var(k_lev[:, 0]))}."
