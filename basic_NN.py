import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import load_data


#Load data
inputs, outputs = load_data()
nb_inputs = 7
nb_outputs = 5

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, stratify=outputs, random_state=1)

# Hyperparameters
learning_rate = 1.59e-7
momentum = 0.95
nb_epochs = 10
batch_size = 128

nb_neurone_layer_1 = 256
nb_neurone_layer_2 = 256
nb_neurone_layer_3 = 64

mod = tf.Sequential()
mod.add(tf.layers.Dense(nb_neurone_layer_1, activation='relu', input_shape=(nb_inputs,)))
mod.add(tf.layers.Dense(nb_neurone_layer_2, activation='relu'))
mod.add(tf.layers.Dropout(0.5))
mod.add(tf.layers.Dense(nb_neurone_layer_3, activation='relu'))
mod.add(tf.layers.Dense(nb_outputs, activation='linear'))

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=output_layer, labels=y))

# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_layer, y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# optimizer = tf.train.GradientDescentOptimizer(
learning_rate = learning_rate).minimize(cost)

# Plot settings
avg_set = []
epoch_set = []

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

# Training cycle
for epoch in range(nb_epochs):
    avg_cost = 0.
total_batch = int(mnist.train.num_examples / batch_size)

# Loop over all batches
for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# Fit training using batch data sess.run(optimizer, feed_dict = {
x: batch_xs, y: batch_ys})
# Compute average loss
avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
# Display logs per epoch step
if epoch % display_step == 0:
    print
Epoch: ", '%04d' % (epoch + 1), "
cost = ", "
{: .9
f}".format(avg_cost)
avg_set.append(avg_cost)
epoch_set.append(epoch + 1)
print
"Training phase finished"

plt.plot(epoch_set, avg_set, 'o', label='MLP Training phase')
plt.ylabel('cost')
plt.xlabel('epoch')
plt.legend()
plt.show()

# Test model
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print
"Model Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})