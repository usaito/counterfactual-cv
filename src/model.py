"""Weighted CounterFactual Regression model for coutnerfactual closs validation.
ref: https://github.com/d909b/perfect_match/tree/master/perfect_match/models/baselines/cfr
CFR paper: https://arxiv.org/abs/1606.03976
"""

import numpy as np
import tensorflow as tf


class CFR:
    """CounterFactual Regression model."""

    def __init__(
            self,
            hidden_layer_size: int,
            num_layers: int,
            learning_rate: float = 0.001,
            batch_size: int = 128,
            dropout: float = 0.,
            imbalance_loss_weight: float = 1.0,
            l2_weight: float = 0.01):
        self.variables = {}
        self.sess = tf.Session()
        self.weight_decay_loss = 0
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.imbalance_loss_weight_ = imbalance_loss_weight
        self.l2_weight_ = l2_weight
        tf.set_random_seed(12345)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.weight_decay_loss += wd * tf.nn.l2_loss(var)
        return var

    def _build_graph(self, input_dim: int, hidden_layer_size: int,
                     num_representation_layers: int, num_regression_layers: int,
                     wass_lambda: float = 10.0, wass_iterations: int = 10, wass_bpt: bool = True):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):
        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """
        ''' Initialize input placeholders '''
        self.x = tf.placeholder("float", shape=[None, input_dim], name='x')
        self.t = tf.placeholder("float", shape=[None, 1], name='t')
        self.y_ = tf.placeholder("float", shape=[None, 1], name='y_')

        ''' Parameter placeholders '''
        self.imbalance_loss_weight = tf.placeholder("float", name='r_alpha')
        self.l2_weight = tf.placeholder("float", name='r_lambda')
        self.dropout_representation = tf.placeholder(
            "float", name='dropout_in')
        self.dropout_regression = tf.placeholder("float", name='dropout_out')

        dim_input = input_dim
        dim_in = hidden_layer_size
        dim_out = hidden_layer_size

        weights_in, biases_in = [], []

        if num_representation_layers == 0:
            dim_in = dim_input
        if num_regression_layers == 0:
            dim_out = dim_in

        ''' Construct input/representation layers '''
        h_rep, weights_in, biases_in = build_mlp(self.x, num_representation_layers, dim_in,
                                                 self.dropout_representation)

        # Normalize representation.
        h_rep_norm = h_rep / \
            safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))

        ''' Construct ouput layers '''
        y, y_concat, weights_out, weights_pred = self._build_output_graph(h_rep_norm, self.t, dim_in, dim_out,
                                                                          self.dropout_regression,
                                                                          num_regression_layers)

        ''' Compute sample reweighting '''
        w_t = self.t / tf.reduce_mean(self.t)
        w_c = (1 - self.t) / (1 - tf.reduce_mean(self.t))
        sample_weight = w_t + w_c
        ''' Construct factual loss function '''
        risk = tf.reduce_mean(sample_weight * tf.square(self.y_ - y))
        pred_error = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - y)))

        ''' Regularization '''
        for i in range(0, num_representation_layers):
            self.weight_decay_loss += tf.nn.l2_loss(weights_in[i])

        p_ipm = 0.5
        imb_dist, imb_mat = wasserstein(h_rep_norm, self.t, p_ipm, sq=True,
                                        its=wass_iterations, lam=wass_lambda, backpropT=wass_bpt)
        imb_error = self.imbalance_loss_weight * imb_dist

        ''' Total error '''
        tot_error = risk + imb_error
        tot_error += self.l2_weight * self.weight_decay_loss

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

    def _build_output(self, h_input: int, dim_in: int, dim_out: int,
                      dropout_regression: float, num_regression_layers: int):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out] * num_regression_layers)

        weights_out = []
        biases_out = []

        for i in range(0, num_regression_layers):
            wo = self._create_variable_with_weight_decay(
                tf.random_normal([dims[i], dims[i + 1]],
                                 stddev=0.1 / np.sqrt(dims[i])),
                'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1, dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

            h_out.append(tf.nn.relu(z))
            h_out[i + 1] = tf.nn.dropout(h_out[i + 1],
                                         1.0 - dropout_regression)

        weights_pred = self._create_variable(tf.random_normal([dim_out, 1], stddev=0.1 / np.sqrt(dim_out)),
                                             'w_pred')

        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        self.weight_decay_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dropout_representation, dropout_regression,
                            num_regression_layers):
        ''' Construct output/regression layers '''
        i0, i1 = tf.to_int32(tf.where(t < 1)[:, 0]), tf.to_int32(
            tf.where(t > 0)[:, 0])
        rep0, rep1 = tf.gather(rep, i0), tf.gather(rep, i1)

        y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in,
                                                             dropout_representation, dropout_regression,
                                                             num_regression_layers)
        y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in,
                                                             dropout_representation, dropout_regression,
                                                             num_regression_layers)

        y = tf.dynamic_stitch([i0, i1], [y0, y1])
        weights_out = weights_out0 + weights_out1
        weights_pred = weights_pred0 + weights_pred1

        y_concat = tf.concat([y0, y1], axis=0)

        return y, y_concat, weights_out, weights_pred

    def train(self, x: np.ndarray, t: np.ndarray, y: np.ndarray,
              num_epochs: int = 200, learning_rate_decay: float = 0.95, iterations_per_decay: int = 100):

        self._build_graph(x.shape[1], self.hidden_layer_size,
                          num_representation_layers=self.num_layers,
                          num_regression_layers=self.num_layers,)

        global_step = tf.Variable(0, trainable=False, dtype="int64")

        lr = tf.train.exponential_decay(self.learning_rate, global_step,
                                        iterations_per_decay, learning_rate_decay, staircase=True)

        opt = tf.train.AdamOptimizer(lr)
        train_step = opt.minimize(self.tot_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())

        np.random.seed(12345)
        for epoch_idx in range(num_epochs):
            idx = np.random.choice(
                np.arange(self.batch_size), size=self.batch_size)
            feed_dict = {
                self.x: x[idx],
                self.t: np.expand_dims(t[idx], 1),
                self.y_: np.expand_dims(y[idx], 1),
                self.dropout_regression: self.dropout,
                self.dropout_representation: self.dropout,
                self.imbalance_loss_weight: self.imbalance_loss_weight_,
                self.l2_weight: self.l2_weight_,
            }

            self.sess.run(train_step, feed_dict=feed_dict)

    def predict(self, x: np.ndarray):
        mu0_pred = self.sess.run(self.output,
                                 feed_dict={
                                     self.x: x,
                                     self.t: np.zeros((x.shape[0], 1)),
                                     self.dropout_representation: 0.0,
                                     self.dropout_regression: 0.0,
                                     self.l2_weight: 0.0
                                 })

        mu1_pred = self.sess.run(self.output,
                                 feed_dict={
                                     self.x: x,
                                     self.t: np.ones((x.shape[0], 1)),
                                     self.dropout_representation: 0.0,
                                     self.dropout_regression: 0.0,
                                     self.l2_weight: 0.0
                                 })

        return mu0_pred.flatten(), mu1_pred.flatten()


class WeightedCFR:
    """Weighted CounterFactual Regression model."""

    def __init__(
            self,
            hidden_layer_size: int,
            num_layers: int,
            learning_rate: float = 0.001,
            batch_size: int = 128,
            dropout: float = 0.,
            imbalance_loss_weight: float = 1.0,
            l2_weight: float = 0.01):
        self.variables = {}
        self.sess = tf.Session()
        self.weight_decay_loss = 0
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.imbalance_loss_weight_ = imbalance_loss_weight
        self.l2_weight_ = l2_weight
        tf.set_random_seed(12345)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.weight_decay_loss += wd * tf.nn.l2_loss(var)
        return var

    def _build_graph(self, input_dim: int, hidden_layer_size: int,
                     num_representation_layers: int, num_regression_layers: int,
                     wass_lambda: float = 10.0, wass_iterations: int = 10, wass_bpt: bool = True):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):
        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """
        ''' Initialize input placeholders '''
        self.x = tf.placeholder("float", shape=[None, input_dim], name='x')
        self.t = tf.placeholder("float", shape=[None, 1], name='t')
        self.e = tf.placeholder("float", shape=[None, 1], name='e')
        self.y_ = tf.placeholder("float", shape=[None, 1], name='y_')

        ''' Parameter placeholders '''
        self.imbalance_loss_weight = tf.placeholder("float", name='r_alpha')
        self.l2_weight = tf.placeholder("float", name='r_lambda')
        self.dropout_representation = tf.placeholder(
            "float", name='dropout_in')
        self.dropout_regression = tf.placeholder("float", name='dropout_out')

        dim_input = input_dim
        dim_in = hidden_layer_size
        dim_out = hidden_layer_size

        weights_in, biases_in = [], []

        if num_representation_layers == 0:
            dim_in = dim_input
        if num_regression_layers == 0:
            dim_out = dim_in

        ''' Construct input/representation layers '''
        h_rep, weights_in, biases_in = build_mlp(self.x, num_representation_layers, dim_in,
                                                 self.dropout_representation)

        # Normalize representation.
        h_rep_norm = h_rep / \
            safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))

        ''' Construct ouput layers '''
        y, y_concat, weights_out, weights_pred = self._build_output_graph(h_rep_norm, self.t, dim_in, dim_out,
                                                                          self.dropout_regression,
                                                                          num_regression_layers)

        ''' Compute sample reweighting '''
        w_t = self.t * (1 - self.e) / self.e
        w_c = (1 - self.t) * self.e / (1 - self.e)
        sample_weight = tf.clip_by_value(
            w_t + w_c, clip_value_min=0.0, clip_value_max=100)
        ''' Construct factual loss function '''
        risk = tf.reduce_mean(sample_weight * tf.square(self.y_ - y))
        pred_error = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - y)))

        ''' Regularization '''
        for i in range(0, num_representation_layers):
            self.weight_decay_loss += tf.nn.l2_loss(weights_in[i])

        p_ipm = 0.5
        imb_dist, imb_mat = wasserstein(h_rep_norm, self.t, p_ipm, sq=True,
                                        its=wass_iterations, lam=wass_lambda, backpropT=wass_bpt)
        imb_error = self.imbalance_loss_weight * imb_dist

        ''' Total error '''
        tot_error = risk + imb_error
        tot_error += self.l2_weight * self.weight_decay_loss

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

    def _build_output(self, h_input: int, dim_in: int, dim_out: int,
                      dropout_regression: float, num_regression_layers: int):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out] * num_regression_layers)

        weights_out = []
        biases_out = []

        for i in range(0, num_regression_layers):
            wo = self._create_variable_with_weight_decay(
                tf.random_normal([dims[i], dims[i + 1]],
                                 stddev=0.1 / np.sqrt(dims[i])),
                'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1, dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

            h_out.append(tf.nn.relu(z))
            h_out[i + 1] = tf.nn.dropout(h_out[i + 1],
                                         1.0 - dropout_regression)

        weights_pred = self._create_variable(tf.random_normal([dim_out, 1], stddev=0.1 / np.sqrt(dim_out)),
                                             'w_pred')

        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        self.weight_decay_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dropout_representation, dropout_regression,
                            num_regression_layers):
        ''' Construct output/regression layers '''
        i0, i1 = tf.to_int32(tf.where(t < 1)[:, 0]), tf.to_int32(
            tf.where(t > 0)[:, 0])
        rep0, rep1 = tf.gather(rep, i0), tf.gather(rep, i1)

        y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in,
                                                             dropout_representation, dropout_regression,
                                                             num_regression_layers)
        y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in,
                                                             dropout_representation, dropout_regression,
                                                             num_regression_layers)

        y = tf.dynamic_stitch([i0, i1], [y0, y1])
        weights_out = weights_out0 + weights_out1
        weights_pred = weights_pred0 + weights_pred1

        y_concat = tf.concat([y0, y1], axis=0)

        return y, y_concat, weights_out, weights_pred

    def train(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, e: np.ndarray,
              num_epochs: int = 200, learning_rate_decay: float = 0.95, iterations_per_decay: int = 100):

        self._build_graph(x.shape[1], self.hidden_layer_size,
                          num_representation_layers=self.num_layers,
                          num_regression_layers=self.num_layers,)

        global_step = tf.Variable(0, trainable=False, dtype="int64")

        lr = tf.train.exponential_decay(self.learning_rate, global_step,
                                        iterations_per_decay, learning_rate_decay, staircase=True)

        opt = tf.train.AdamOptimizer(lr)
        train_step = opt.minimize(self.tot_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())

        np.random.seed(12345)
        for epoch_idx in range(num_epochs):
            idx = np.random.choice(
                np.arange(self.batch_size), size=self.batch_size)
            feed_dict = {
                self.x: x[idx],
                self.t: np.expand_dims(t[idx], 1),
                self.e: np.expand_dims(e[idx], 1),
                self.y_: np.expand_dims(y[idx], 1),
                self.dropout_regression: self.dropout,
                self.dropout_representation: self.dropout,
                self.imbalance_loss_weight: self.imbalance_loss_weight_,
                self.l2_weight: self.l2_weight_,
            }

            self.sess.run(train_step, feed_dict=feed_dict)

    def predict(self, x: np.ndarray):
        mu0_pred = self.sess.run(self.output,
                                 feed_dict={
                                     self.x: x,
                                     self.t: np.zeros((x.shape[0], 1)),
                                     self.dropout_representation: 0.0,
                                     self.dropout_regression: 0.0,
                                     self.l2_weight: 0.0
                                 })

        mu1_pred = self.sess.run(self.output,
                                 feed_dict={
                                     self.x: x,
                                     self.t: np.ones((x.shape[0], 1)),
                                     self.dropout_representation: 0.0,
                                     self.dropout_regression: 0.0,
                                     self.l2_weight: 0.0
                                 })

        return mu0_pred.flatten(), mu1_pred.flatten()


def build_mlp(x: np.ndarray, num_layers: int = 1, hidden_layer_size: int = 16, dropout: float = 0.0):
    input_dim = int(x.shape[-1])
    h_in, weights_in, biases_in = [x], [], []
    for i in range(0, num_layers):
        if i == 0:
            ''' If using variable selection, first layer is just rescaling'''
            weights_in.append(tf.Variable(tf.random_normal([input_dim, hidden_layer_size],
                                                           stddev=0.1 / np.sqrt(input_dim))))
        else:
            weights_in.append(tf.Variable(tf.random_normal([hidden_layer_size, hidden_layer_size],
                                                           stddev=0.1 / np.sqrt(hidden_layer_size))))

        biases_in.append(tf.Variable(tf.zeros([1, hidden_layer_size])))
        z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

        h_in.append(tf.nn.relu(z))
        h_in[i + 1] = tf.nn.dropout(h_in[i + 1], 1.0 - dropout)

    h_rep = h_in[len(h_in) - 1]
    return h_rep, weights_in, biases_in


def wasserstein(X, t, p, lam=1, its=50, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it, ic = tf.where(t > 0)[:, 0], tf.where(t < 1)[:, 0]
    Xc, Xt = tf.gather(X, ic), tf.gather(X, it)
    nc, nt = tf.to_float(tf.shape(Xc)[0]), tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2sq(Xt, Xc) if sq else safe_sqrt(pdist2sq(Xt, Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M, 10 / (nc * nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam / M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat(
        [delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))], 0)
    Mt = tf.concat([M, row], 0)
    Mt = tf.concat([Mt, col], 1)

    ''' Compute marginal vectors '''
    a = tf.concat([p * tf.ones(tf.shape(tf.where(t > 0)[:, 0:1])
                               ) / nt, (1 - p) * tf.ones((1, 1))], 0)
    b = tf.concat([(1 - p) * tf.ones(tf.shape(tf.where(t < 1)
                                              [:, 0:1])) / nc, p * tf.ones((1, 1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / \
            (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
    v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

    T = u * (tf.transpose(v) * K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T * Mt
    D = 2 * tf.reduce_sum(E)

    return D, Mlam


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def pdist2(X, Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X, Y))


def pop_dist(X, t):
    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt, Xc)
    return M


def safe_sqrt(x, lbound=1e-10):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))
