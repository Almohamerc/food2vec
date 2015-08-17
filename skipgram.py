import numpy as np


# We use the skip-gram model without negative sampling
# reference: http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf

# Initialization of W is taken from the reference implementation:
# https://groups.google.com/forum/#!topic/word2vec-toolkit/du6teCT5Pug

# Code reference:
# https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py

# linearly reduce this to 0
LR = 0.025
NUM_EPOCHS = 10
K = 2

def build_indices(threshold=5):
    "Return a dict.  Dict assigns a number to each word (if count exceeded)."
    indices = {}
    counts = {}
    i = 0
    for line in open('data/srep00196-s3.csv','r'):
        # skip header
        if '#' not in line:
            words = line.rstrip().split(',')            
            for word in words:
                if word not in counts:
                    counts[word] = 0
                else:
                    counts[word] += 1
                if counts[word] >= threshold and word not in indices:
                    indices[word] = i
                    i = i + 1
    return (indices, counts)

def line_generator(indices):
    "Generator, yield list of word indices for each line in data."
    for line in open('data/srep00196-s3.csv','r'):
        # skip headers
        if '#' not in line:
            tokens = line.rstrip().split(',')            
            yield [indices[t] for t in tokens if t in indices]


if __name__ == '__main__':
    (indices, counts) = build_indices()
    N = len(indices)
    W = np.random.uniform(low=-0.5/K, high=0.5/K, size=(N, K))

    # use each food (food_x) to predict others (food_y) in the same recipe
    for epoch in xrange(NUM_EPOCHS):        
        total_loss = 0.0 # per line
        num_lines = 0
        for recipe in line_generator(indices):
            for food_x in recipe:
                # calculate softmax(X*W*W^T) for food_x
                output = 1. /(1. + np.exp(-np.dot(W[food_x], W.T)))
                target = np.zeros(N)
                target[recipe] = 1.
                target[food_x] = 0.

                # calculate error wrt predicting food_y
                grad = output - target
                # first W update (hidden->output)
                W -= LR * np.outer(grad, W[food_x])
                # input->hidden update
                W[food_x] -= LR * np.dot(grad, W)

                total_loss += np.mean(grad)
    
    # print result
    i = 0
    i2word = {v: k for k, v in indices.items()}
    for row in W:
        tokens = [i2word[i]]
        tokens.extend(row.tolist())
        tokens.append(counts[i2word[i]])
        print ','.join([str(t) for t in tokens])
        i += 1
