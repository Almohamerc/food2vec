with open('food2vec.csv', 'rb') as f:
    # skip first line
    f.readline()
    for line in f.readlines():
        tokens = line.strip().split(',')
        hot_indices = [i for i, x in enumerate(tokens) if x=='1']
        for i in hot_indices:
            # x is one hot
            x_vector = [0] * len(tokens)
            x_vector[i] = 1
            # print x vector to X headerless csv file
            y_vector = [int(t) for t in tokens]
            y_vector[i] = 0
            # print y vector to Y file
            import pdb; pdb.set_trace()
