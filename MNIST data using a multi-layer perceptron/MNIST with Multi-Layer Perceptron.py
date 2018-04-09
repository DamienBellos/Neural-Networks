#  MNIST data set of handwritten digits from (http://yann.lecun.com/exdb/mnist/).
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Data Format
type(mnist)
type(mnist.train.images)

# mnist.train.images[0]
mnist.train.images[2].shape
sample = mnist.train.images[2].reshape(28, 28)
plt.imshow(sample)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
n_samples = mnist.train.num_examples

#  TensorFlow Graph Input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# MultiLayer Model
def multilayer_perceptron(x, weights, biases):
    # First Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Second Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Last Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Weights and Bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialization of Variables
init = tf.initialize_all_variables()

# Training the Model
# next_batch()
Xsamp, ysamp = mnist.train.next_batch(1)
plt.imshow(Xsamp.reshape(28, 28))
print(ysamp)

# Launch the session
sess = tf.InteractiveSession()

# Intialize all the variables
sess.run(init)

# Training Epochs
for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch
    print("Epoch: {} cost={:.4f}".format(epoch + 1, avg_cost))
print("Model has completed {} Epochs of Training".format(training_epochs))

# Model Evaluations
correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
print(correct_predictions[0])
correct_predictions = tf.cast(correct_predictions, "float")
print(correct_predictions[0])

# Select the mean of the elements across the tensor.
accuracy = tf.reduce_mean(correct_predictions)
type(accuracy)

# Call the MNIST test labels and images and evaluate the accuracy!
mnist.test.labels
mnist.test.images
print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
