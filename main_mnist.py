import tensorflow as tf
import sonnet as snt

import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler

from BNN_MLP import *


BATCH_SIZE = 64
TRAINING_STEPS = 20001


mnist = tf.contrib.learn.datasets.load_dataset("mnist")

def make_tf_data_batch(x, y, shuffle=True):
    # create Dataset objects using the data previously downloaded
    dataset_train = tf.data.Dataset.from_tensor_slices((x, y.astype(np.int32)))

    if shuffle:
        dataset_train = dataset_train.shuffle(100000)

    # we shuffle the data and sample repeatedly batches for training
    batched_dataset_train = dataset_train.repeat().batch(BATCH_SIZE)
    # create iterator to retrieve batches
    iterator_train = batched_dataset_train.make_one_shot_iterator()
    # get a training batch of images and labels
    (batch_train_images, batch_train_labels) = iterator_train.get_next()

    return batch_train_images, batch_train_labels


train_x, train_y = make_tf_data_batch(mnist.train.images, mnist.train.labels)
test_x, test_y = make_tf_data_batch(mnist.test.images, mnist.test.labels, shuffle=False)



# Initialize the network and optimizer
net = BNN_MLP(n_inputs=784, n_outputs=10, hidden_units=[200, 200], init_mu=0.0, init_rho=-3.0, activation=tf.nn.relu, last_activation=tf.identity)

out, log_probs, nll = net(train_x, targets=train_y, sample=True, n_samples=1, loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y))

num_batches = (len(mnist.train.labels)//BATCH_SIZE)
loss = 0.01*log_probs/num_batches + nll


optim = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optim.minimize( loss )


# Test accuracy
out_test_deterministic, _, _ = net(test_x, sample=False, loss_function=None)
prediction = tf.cast(tf.argmax(out_test_deterministic, 1), tf.int32)
equality = tf.equal(prediction, test_y)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))



sess = tf.InteractiveSession()
sess.run( tf.global_variables_initializer() )


for i in range(TRAINING_STEPS):
    l, _ = sess.run([loss, train_op])

    if i>=1000 and i%1000==0:
        # Test accuracy
        avg_acc = 0.0
        num_iters = len(mnist.test.labels)//BATCH_SIZE
        for test_iter in range(num_iters):
            acc = sess.run(accuracy)
            avg_acc += acc

        avg_acc /= num_iters
        print("Iteration ", i, "loss: ", l, "accuracy: ", avg_acc)


        ## Histogram of standard deviations (w and b)
        all_stds = []
        for l in net.layers:
            w_sigma = np.reshape( sess.run(l.w_sigma), [-1] ).tolist()
            b_sigma = np.reshape( sess.run(l.b_sigma), [-1] ).tolist()
            all_stds += w_sigma + b_sigma

        n = TRAINING_STEPS//1000
        plt.rc('axes', prop_cycle=cycler('color', [plt.get_cmap('inferno')(1. * float(i)/n) for i in range(n)]))
        lbl = ""
        if i==1000:
            lbl = "t=1000"
        elif i==1000*(TRAINING_STEPS//1000):
            lbl = "t="+str(1000*(TRAINING_STEPS//1000))
        plt.hist(all_stds, 100, alpha=0.3, label=lbl)

plt.legend()
plt.savefig("mnist_w_sigma_hist.png")
plt.show()



