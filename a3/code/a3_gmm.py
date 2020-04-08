from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import pickle as pkl
import sys
dataDir = '/u/cs401/A3/data/'


def logexp(a, b):
  return np.log(np.exp(a-b).sum(axis=0))


# def precomputed(myTheta, m):
#     n, d = myTheta.mu.shape
#     sig = myTheta.sigma


def compute_logs(x, M, theta):
    log_bs = np.array([log_b_m_x(m, x, theta) for m in range(M)])
    log_ps = [log_p_m_x(m, x, theta, log_bs) for m in range(M)]
    return log_bs, np.array(log_ps)


def update_theta(theta, x, log_pms):
    M = len(theta.omega)
    T = len(x)
    # p(m | x_t ; theta)
    pms = np.exp(log_pms)

    # update the probabilities of each mixture
    theta.omega = (pms.sum(axis=1) / T).reshape(M, 1)
    norm = T * theta.omega
    # update Gaussian means
    theta.mu = (pms @ x) / norm

    # update variances
    theta.sigma = (pms @ np.power(x, 2)) / norm - np.power(theta.mu, 2)

    return theta


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    # if len(preComputedForM) == 0:
    #   preComputedForM = precomputed(myTheta, m)
    mu = myTheta.mu[m]  # mean of m
    d = len(mu)  # d-dimensional vector
    sig = myTheta.Sigma[m]  # sigma of m
    term1 = 0.5 * (np.power(x - mu, 2) / sig).sum(axis=1)
    term2 = d/2 * np.log(2 * np.pi)
    term3 = 0.5 * np.log(sig).sum()
    return -term1 - term2 - term3

    
def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    n = len(myTheta.omega)
    l_omg = np.log(myTheta.omega)
    term1 = np.log(myTheta.omega[m])
    term2 = log_b_m_x(m, x, myTheta)
    alllog = np.add([log_b_m_x(i, x, myTheta) for i in range(n)], l_omg)
    term3 = alllog.max() + logexp(alllog, alllog.max())
    return term1 + term2 - term3

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    alllog = log_Bs + np.log(myTheta.omega)
    return (alllog.max(axis=0) + logexp(alllog, alllog.max(axis=0))).sum()

    
def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])

    i = 0
    prev_l = -np.inf
    delta = np.inf
    while i < maxIter and delta >= epsilon:
        log_bs, log_ps = compute_logs(X, M, myTheta)
        l = logLik(log_bs, myTheta)
        myTheta = update_theta(theta, X, log_ps)
        delta = prev_l - l
        prev_l = l
        i+= 1
        print(f"iteration {i} done with l: {round(l, 3)} and delta: {round(delta, 3)}")
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    loglikes = []
    M = len(models[0].omega)
    for i, model in enumerate(models):
        log_bs, log_ps = compute_logs(mfcc, M, model)
        l = logLik(log_bs, model)
        loglikes.append(l)
    best_indices = np.argsort(-np.array(loglikes))  # -'ves to reverse order
    bestModel = best_indices[0]
    if k > 0:
        print(models[correctID].name)
        top_k = best_indices[:k]
        kmodels = [models[i] for i in top_k]
        klogs = [loglikes[i] for i in top_k]
        for model, loglike in zip(kmodels, klogs):
            print(f"{model} {loglike}")
        print("")
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    train_xs = []
    f = "saved.pkl"
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    retry = True
    try:
        if not retry:  # see if we want to load. If we don't retry is True.
          with open(f, "rb") as infile:
              trainThetas = pkl.load(infile)
              testMFCCs = pkl.load(infile)
              train_xs = pkl.load(infile)
          print("loading successful")
    except:  # didn't work so retrty
        retry = True
    if retry:
      i = 0
      # train a model for each speaker, and reserve data for testing
      for subdir, dirs, files in os.walk(dataDir):
          for speaker in dirs:
              print(speaker)

              files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
              random.shuffle(files)

              testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
              testMFCCs.append(testMFCC)

              X = np.empty((0, d))
              for file in files:
                  myMFCC = np.load(os.path.join(dataDir, speaker, file))
                  X = np.append(X, myMFCC, axis=0)

              trainThetas.append(train(speaker, X, M, epsilon, maxIter))
              train_xs.append(X)
              i += 1
              print(f"completed {i} of {len(dirs)}")

    if retry:  # we retried and got new results, so save them
        with open(f, 'wb') as outfile:
            pkl.dump(trainThetas, outfile)
            pkl.dump(testMFCCs, outfile)
            pkl.dump(train_xs, outfile)
    stdout = sys.stdout  # steal stdout so that we can redirect to file.
    sys.stdout = open('gmmLiks.txt', 'w')
    # evaluate
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0*numCorrect/len(testMFCCs)
    sys.stdout = stdout
    print(f"accuracy: {accuracy}")
