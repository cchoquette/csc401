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
    R = np.zeros((len(r) + 2, len(h) + 2))
    R[0, :] = np.arange(R.shape[1])
    R[:, 0] = np.arange(R.shape[0])
    # R[0, :, 1] = np.ones_like(R[0, :, 1])  # inserts
    # R[:, 0, 1] = np.zeros_like(R[:, 0, 1])  # deletes
    # now setup strings for comparison
    r = add_tags(r)
    h = add_tags(h)
    # following the forward-backward algorithm, this is the forward part.
    for i in range(R.shape[0]-1):
        for j in range(R.shape[1]-1):
            j_1 = j+1
            i_1 = i+1
            same = int(r[i_1] == h[j_1])
            delete = R[i, j_1]
            insert = R[i_1, j]
            substitute = R[i, j] - same  # if they were the same, will cancel with the +1 later
            choices = np.array([delete, insert, substitute]) + 1
            # R[i_1, j_1, 1] = choices.argmin()
            R[i_1, j_1] = choices.min()
    print(R)
    # now we do the backward algorithm
    counts = {0: 0, 1: 0, 2: 0}  # indices correspond to choices
    i, j = R.shape
    i -= 1
    j -= 1
    while i > 0 or j > 0:
        i_1 = i - 1
        j_1 = j - 1
        if i > 0 and R[i_1, j] == R[i, j] - 1:
            counts[0] += 1  # deletion
            i -= 1
        elif j > 0 and R[i, j_1] == R[i, j] - 1:
            counts[1] += 1  # insertion
            j -= 1
        elif i > 0 and j > 0 and R[i_1, j_1] == R[i, j] - 1:
            counts[2] += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and R[i_1, j_1] == R[i, j]:
            i -= 1
            j -= 1
        # match_type = R[i, j, 1]
        # # extra check to see if its a match or a substitute
        # to_add = 0 if match_type == 2 and r[i-1] == h[i-1] else 1
        # counts[match_type] += to_add
        # i_dec = 0 if match_type == 1 else 1  # we don't decrement i if insert
        # j_dec = 0 if match_type == 0 else 1  # we don't decrement j if delete
        # i -= i_dec
        # j -= j_dec

    # return, WER, followed by counts, mine are in reverse order
    return [R[-1, -1] / (len(R) - 2)] + [counts[i] for i in reversed(range(3))]


def process_line(l):
    return re.sub(r"[^a-z\[\] ]+", "", l.lower()).split()


def make_print(speaker, t, i, out):
    beg = [speaker, t, i, out[0]]
    prefixes = ['S:{}', 'I:{}', 'D:{}']
    beg + list([p.format(o) for p, o in zip(prefixes, out[1:])])
    print(beg)
    return beg


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
            for i, (r, g, k) in enumerate(alllines):
                r = process_line(r)
                print(r)
                print(process_line(g))
                print(process_line(k))
                goog = Levenshtein(r, process_line(g))
                g_lev.append(goog)
                gog = make_print(speaker, 'Google', i, goog)
                kald = Levenshtein(r, process_line(k))
                k_lev.append(kald)
                kald = make_print(speaker, 'Kaldi', i, kald)
                s = " ".join(["{}" for _ in range(7)])
                print(s.format(*goog))
                print(s.format(*kald))
    g_lev = np.array(g_lev)
    k_lev = np.array(k_lev)
    out1 = f"Google has a mean of: {np.mean(g_lev[:, 0])} " \
        f"and std_dev of: {np.sqrt(np.var(g_lev[:, 0]))}. Kaldi has a mean of: " \
        f"{np.mean(k_lev[:, 0])} and std_dev of: " \
        f"{np.sqrt(np.var(k_lev[:, 0]))}."
