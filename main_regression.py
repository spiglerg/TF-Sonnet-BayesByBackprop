import tensorflow as tf
import sonnet as snt

import numpy as np

import matplotlib.pyplot as plt

from BNN_MLP import *


from random import shuffle

test_n_samples = 100

TRAINING_STEPS = 4000


# Build the training set
x = np.random.uniform(-4, 4, size=20).reshape((-1, 1))
noise = np.random.normal(0, 9, size=20).reshape((-1, 1))
y = x ** 3 + noise

x_ = np.linspace(-6, 6, 200)
y_ = x_ ** 3



# Initialize the network and optimizer
net = BNN_MLP(n_inputs=1, n_outputs=1, hidden_units=[100], init_mu=0.0, init_rho=0.0, activation=tf.nn.relu, last_activation=tf.identity)

x_placeholder = tf.placeholder(tf.float32, (None,1))
y_placeholder = tf.placeholder(tf.float32, (None,1))


out, log_probs, nll = net(x_placeholder, targets=y_placeholder, sample=True, n_samples=1, loss_function=lambda y, y_target: 0.5*tf.reduce_sum(tf.square(y_target-y), 1) )

loss = log_probs/40 + nll

optim = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optim.minimize( loss )




sess = tf.InteractiveSession()
sess.run( tf.global_variables_initializer() )


for i in range(TRAINING_STEPS):
    avg_loss, _ = sess.run([loss, train_op], feed_dict={x_placeholder:x, y_placeholder:y})

    if i%100 == 0:
        print('Iteration ', i, "loss: ", avg_loss)


# Sample test_n_samples networks and test them at every point, to compute the average prediction and standard deviation
ys = []
for i in range(test_n_samples):
    ys.append( sess.run( out, feed_dict={x_placeholder:np.expand_dims(x_,-1)} ) )
ys = np.asarray(ys)

plt.plot(x_, np.mean(ys,0), c='royalblue', label='mean pred')
plt.fill_between(x_, np.squeeze(np.mean(ys,0) - 3*np.std(ys,0)), np.squeeze(np.mean(ys,0) + 3*np.std(ys,0)), color='cornflowerblue', alpha=.5, label='+/- 3 std')
plt.plot( x, y, '*', color='black', label='training data' )

plt.legend()
plt.tight_layout()
plt.savefig("regression.png")
plt.show()



"""
all_stds = []
for l in net.layers:
    w_sigma = np.reshape( sess.run(l.w_sigma), [-1] ).tolist()
    b_sigma = np.reshape( sess.run(l.b_sigma), [-1] ).tolist()
    all_stds += w_sigma + b_sigma

plt.hist(all_stds, 100)
plt.show()
"""








