import tensorflow as tf
import sonnet as snt

import numpy as np

from BNNLayer import *


class BNN_MLP(snt.AbstractModule):
    """
    Implementation of an Bayesian MLP with structure [n_inputs] -> hidden_units -> [n_outputs], with a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self, n_inputs, n_outputs, hidden_units=[], init_mu=0.0, init_rho=0.0, activation=tf.nn.relu, last_activation=tf.nn.softmax, name="BNN_MLP"):
        super(BNN_MLP, self).__init__(name=name)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.last_activation = last_activation
        hidden_units = [n_inputs] + hidden_units + [n_outputs]

        self.layers = []
        for i in range(1, len(hidden_units)):
            self.layers.append( BNNLayer(hidden_units[i-1], hidden_units[i], init_mu=init_mu, init_rho=init_rho) )


    def _build(self, inputs, sample=False, n_samples=1, targets=None, loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y) ):
        """
        Constructs the MLP graph.

        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value
          n_samples: number of sampled networks to average output of the MLP over
          targets: target outputs of the MLP, used to compute the loss function on each sampled network
          loss_function: lambda function to compute the loss of the network given its output and targets.

        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
          avg_loss: `tf.Tensor` average loss across n_samples, computed using `loss_function'
        """

        log_probs = 0.0
        avg_loss = 0.0

        if not sample:
            n_samples = 1

        output = 0.0 ## avg. output logits
        for ns in range(n_samples):
            x = inputs
            for i in range(len(self.layers)):
                x, l_prob = self.layers[i](x, sample)
                if i == len(self.layers)-1:
                    x = self.last_activation(x)
                else:
                    x = self.activation(x)

                log_probs += l_prob

            output += x

            if targets is not None:
                if loss_function is not None:
                    loss = tf.reduce_mean(loss_function(x, targets), 0)

                    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=x))
                    #loss = 0.5*tf.reduce_mean(tf.reduce_sum( tf.square(targets-x), 1), 0)
                    avg_loss += loss


        log_probs /= n_samples
        avg_loss /= n_samples
        output /= n_samples

        return output, log_probs, avg_loss


