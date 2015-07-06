from scuba2 import *

LR = 0.01
EPOCHS = 50
TESTS_PER_EPOCH = 10
BATCH_SIZE = 20


def test_mlp(n_epochs=1000, shape=None, almost_linear=False):
    numpy.random.seed(12345)

    datasets = load_mnist()
    train_x, train_y = datasets[0]
    valid_set_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    n_train_batches = train_x.get_value(borrow=True).shape[0] / BATCH_SIZE

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    mlp = MLP(input=x)
    mlp.add(Linear(28*28, shape[0]))
    mlp.add(Tanh())
    for i in xrange(len(shape)-1):
        mlp.add(Linear(shape[i], shape[i+1]))
        mlp.add(Tanh())
    mlp.add(Linear(shape[-1], 10))
    mlp.add(Softmax())
    cost = mlp.NLL(y)
    # sanity check: should be 8.728 test_error at epoch 2
    # now 8.204 after separating softmax from linear

    test_model = theano.function(inputs=[], outputs=mlp.errors(y),
                                 givens={x: test_x, y: test_y})
    validate_model = theano.function(inputs=[], outputs=mlp.errors(y),
                                     givens={x: valid_set_x, y: valid_y})
    train_results = theano.function(inputs=[], outputs=[cost, mlp.errors(y)],
                                    givens={x: train_x, y: train_y})

    gparams = [T.grad(cost, param) for param in mlp.params]
    updates = [
        (param, param - LR * gparam)
        for param, gparam in zip(mlp.params, gparams)
    ]

    train = theano.function(
        inputs=[index], outputs=cost, updates=updates,
        givens={
            x: train_x[index * BATCH_SIZE: (index + 1) * BATCH_SIZE],
            y: train_y[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
        }
    )

    print mlp

    examples_seen = 0
    shape_str = '-'.join([str(l) for l in shape])
    model_str = "almost_linear" if almost_linear else "nonlinear"

    for epoch in xrange(1, n_epochs):
        [test_err, val_err, train_loss, train_err] = [0.0, 0.0, 0.0, 0.0]
        for batch_index in xrange(n_train_batches):
            if batch_index % (n_train_batches / TESTS_PER_EPOCH) == 0:
                test_err += test_model() * 100. / TESTS_PER_EPOCH
                val_err += validate_model() * 100. / TESTS_PER_EPOCH
                [t_loss, t_error] = train_results()
                train_err += t_error * 100. / TESTS_PER_EPOCH
                train_loss += t_loss * 100. / TESTS_PER_EPOCH
            train(batch_index)
            examples_seen += BATCH_SIZE
        if epoch % 10 == 0:
            print('"%s","%s",%i,%i,%f,%f,%f,%f' %
                  (model_str,shape_str,epoch, examples_seen,
                   train_loss, train_err, val_err, test_err))

    
if __name__ == '__main__':
    print 'type,shape,epoch,examples_seen,train_loss,train_error,validation_error,test_error'
    test_mlp(n_epochs=EPOCHS, shape=[100], almost_linear=False)
    test_mlp(n_epochs=EPOCHS, shape=[100, 100], almost_linear=False)
