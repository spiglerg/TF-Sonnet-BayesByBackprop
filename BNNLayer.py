import tensorflow as tf
import sonnet as snt

import numpy as np


class BNNLayer(snt.AbstractModule):
    """
    Implementation of a linear Bayesian layer with n_inputs and n_outputs, and a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self, n_inputs, n_outputs, init_mu=0.0, init_rho=0.0, name="BNNLayer"):
        super(BNNLayer, self).__init__(name=name)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.w_mean = tf.Variable(init_mu*tf.ones([self.n_inputs, self.n_outputs]))#tf.random_normal([self.n_inputs, self.n_outputs], 0.0, 0.01))
        self.w_rho = tf.Variable(init_rho*tf.ones([self.n_inputs, self.n_outputs]))#tf.Variable(tf.random_normal([self.n_inputs, self.n_outputs], -4, 0.1)) ## -4 or even -3 are good initializations
        self.w_sigma = tf.log(1.0 + tf.exp(self.w_rho))
        self.w_distr = tf.distributions.Normal(loc=self.w_mean, scale=self.w_sigma)

        self.b_mean = tf.Variable(init_mu*tf.ones([self.n_outputs]))#tf.Variable(tf.random_normal([self.n_outputs], 0.0, 0.01))
        self.b_rho = tf.Variable(init_rho*tf.ones([self.n_outputs]))#tf.Variable(tf.random_normal([self.n_outputs], -4, 0.1))
        self.b_sigma = tf.log(1.0 + tf.exp(self.b_rho))
        self.b_distr = tf.distributions.Normal(loc=self.b_mean, scale=self.b_sigma)

        self.w_prior_distr = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.b_prior_distr = tf.distributions.Normal(loc=0.0, scale=1.0)


        ## No need to limit ourselves to diagonal gaussian priors!

        # Mixture of gaussians
        #tfd = tf.distributions
        #pi = 0.5
        #self.w_prior_distr = tf.contrib.distributions.Mixture(cat=tfd.Categorical(probs=[pi, 1.-pi]), components=[tfd.Normal(loc=0.0, scale=1.0), tfd.Normal(loc=0.0, scale=float(np.exp(-6.0)))])
        #self.b_prior_distr = tf.contrib.distributions.Mixture(cat=tfd.Categorical(probs=[pi, 1.-pi]), components=[tfd.Normal(loc=0.0, scale=1.0), tfd.Normal(loc=0.0, scale=float(np.exp(-6.0)))])


    def _build(self, inputs, sample=False):
        """
        Constructs the graph for the layer.

        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value

        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
        """

        if sample:
            w = self.w_distr.sample()
            b = self.b_distr.sample()
        else:
            w = self.w_mean
            b = self.b_mean

        z = tf.matmul(inputs,w) + b

        log_probs = tf.reduce_sum(self.w_distr.log_prob(w)) + tf.reduce_sum(self.b_distr.log_prob(b)) - tf.reduce_sum(self.w_prior_distr.log_prob(w)) - tf.reduce_sum(self.b_prior_distr.log_prob(b))

        return z, log_probs


