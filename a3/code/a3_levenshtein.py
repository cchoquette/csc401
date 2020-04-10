import os
import numpy as np
import sys
import re

dataDir = '/u/cs401/A3/data/'
dataDir = "/scratch/ssd001/home/cchoquet/csc401/a3/code/data/data/"


def add_tags(str_list):
    """Add start and end tag to list of strings.

    :param str_list: a list of strings
    :return: the same list with a start and end tag pre-pended
    and post-pended, respectively
    """
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
    # add a start and end tag to each list so we can compare them
    r = add_tags(r)
    h = add_tags(h)
    # create and initialize our grids
    # dim 0: for r
    # dim 1: for h
    # types are 0: delete, 1: insert, 2: substitute
    R = np.zeros((len(r), len(h)))
    R[:, 0] = np.arange(R.shape[0])
    R[0, :] = np.arange(R.shape[1])
    # following the forward-backward algorithm, this is the forward part.
    for i in range(R.shape[0] - 1):
        for j in range(R.shape[1] - 1):
            i_1 = i + 1  # next indices
            j_1 = j + 1
            same = int(r[i_1] == h[j_1])  # if we have a potential substitution
            sub = R[i, j] - same  # if they are the same (no sub) this will
            # cancel with the + 1 later.
            delete = R[i, j_1]
            insert = R[i_1, j]
            # the minimum cost is always taken.
            R[i_1, j_1] = np.array([delete, insert, sub]).min() + 1
    counts = {0: 0, 1: 0, 2: 0}  # 0: delete, 1: insert, 2: substitution
    i = R.shape[0] - 1
    j = R.shape[1] - 1
    # now the backward part
    while i > 0 or j > 0:
        i_1, j_1 = i-1, j-1  # next indices
        curr_R = R[i, j]  # common numpy accesses
        next_R = R[i_1, j_1]
        # priorities are sub > insert > delete.
        # check sub
        if i > 0 and j > 0 and (next_R == curr_R or next_R == curr_R - 1):
            i -= 1
            j -= 1
            if next_R == curr_R - 1:
                counts[2] += 1
        # check insert
        elif j > 0 and R[i, j - 1] == curr_R - 1:
            counts[1] += 1
            j -= 1
        # check delete
        elif i > 0 and R[i - 1, j] == curr_R - 1:
            counts[0] += 1
            i -= 1
    # return, WER, followed by counts, mine are in reverse order
    return [R[-1, -1] / (R.shape[0] - 2)] + [counts[i] for i in reversed(range(3))]


def process_line(l):
    """

    :param l: a line as string
    :return: string with all punctuation removed.
    """
    return re.sub(r"[^a-z\[\] ]+", "", l.lower()).split()


def make_print(speaker, t, i, out):
    """ Make Levenshtein output easily printable.

    :param speaker: current speaker
    :param t: type
    :param i: position
    :param out: Levenshtein output
    :return: args for .format
    """
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
            alllines = [open(p, 'r').readlines() for p in paths]
            if any([len(x) == 0 for x in alllines]):
                print("a transcript file was empty or missing.")
                continue
            try:
                alllines = zip(*alllines)
            except:
                print("error encountered")
                continue

            for i, (r, g, k) in enumerate(alllines):
                r = process_line(r)[1:]
                print(r)
                goog = Levenshtein(r, process_line(g)[1:])
                g_lev.append(goog)
                goog = make_print(speaker, 'Google', i, goog)
                kald = Levenshtein(r, process_line(k)[1:])
                k_lev.append(kald)
                kald = make_print(speaker, 'Kaldi', i, kald)
                s = " ".join(["{}" for _ in range(7)])
                print(s.format(*goog))
                print(s.format(*kald))
                sys.stdout.flush()
    g_lev = np.array(g_lev)
    k_lev = np.array(k_lev)
    out1 = f"Google has a mean of: {np.mean(g_lev[:, 0])} " \
        f"and std_dev of: {np.sqrt(np.var(g_lev[:, 0]))}. Kaldi has a mean of: " \
        f"{np.mean(k_lev[:, 0])} and std_dev of: " \
        f"{np.sqrt(np.var(k_lev[:, 0]))}. We see that Kaldi performs a bit " \
        f"better than Google in terms of its mean WER, which is about 30% " \
        f"lower than Google's. This large difference appears to make Kaldi " \
        f"to be signifcantly better than Google, however, which such small " \
        f"WER, we would need to conduct statistical testing to be rigorous. " \
        f"We also see that Kaldi " \
        f"is marginally more consistent, with a smaller standard deviation. "
    print(out1)
    out2 = f"We now further analyze the types of errors made by each system. " \
        f"Comparing the mean substitution rate of Google: {np.mean(g_lev[:, 1])}" \
        f" with Kaldi: {np.mean(k_lev[:, 1])}, we see that Google makes fewer " \
        f"substition errors than Kaldi (only marginally). Comparing the " \
        f"insertion errors, Google has: {np.mean(g_lev[:, 2])} and Kaldi has: " \
        f"{np.mean(k_lev[:, 2])}, where Google is again marginally lower. In " \
        f"fact, google on average makes no insertion errors. However, comparing" \
        f" deletion errors, Google has: {np.mean(g_lev[:, 3])} and Kaldi has: " \
        f"{np.mean(k_lev[:, 3])}, where we see that Google makes " \
        f"substantially more deletion errors. These results are interesting as" \
        f" it could show that Google has prioritize minimizing different types " \
        f"of errors than Kaldi, potentially due to the use cases or perceived" \
        f" naturalness."
