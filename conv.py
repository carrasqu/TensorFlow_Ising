import tensorflow as tf
import input_data
import sys

#"images"
lx=20

#parameters of the network
numberlabels=2
hiddenunits1=400
lamb=0.05 # regularization parameter

# how does the data look like
Ntemp=20
samples_per_T=250
Nord=20

#reading the data 
mnist = input_data.read_data_sets(numberlabels,lx,'txt', one_hot=True)


lamb=0.05 # regularization parameter
mnist = input_data.read_data_sets(numberlabels,lx,'txt', one_hot=True)

print "reading sets ok"

#sys.exit("pare aqui")

# defining weighs and initlizatinon
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

# defining the layers
def layers(x, W,b):
  return tf.nn.sigmoid(tf.matmul(x, W)+b)
         

# defining the model

x = tf.placeholder("float", shape=[None, lx*lx])
y_ = tf.placeholder("float", shape=[None, numberlabels])

#first layer 
#weights and bias
W_1 = weight_variable([lx*lx,hiddenunits1])
b_1 = bias_variable([hiddenunits1])

#Apply a sigmoid

O1 = layers(x, W_1,b_1)

#second layer(output layer in this case)
W_2 = weight_variable([hiddenunits1,numberlabels])
b_2 = bias_variable([numberlabels])

O2=layers(O1, W_2,b_2)
y_conv=O2

#Train and Evaluate the Model

# cost function to minimize
cross_entropy = tf.reduce_sum( -y_*tf.log(y_conv)-(1.0-y_)*tf.log(1.0-y_conv)  )+lamb*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2) )

#defining the optimizer
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#optimizer = tf.train.GradientDescentOptimizer(0.5)
optimizer= tf.train.AdamOptimizer(0.0001)
train_step = optimizer.minimize(cross_entropy)

#predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(10000):

  batch = mnist.train.next_batch(100)

  if i%100 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1]})
    print "step %d, training accuracy %g"%(i, train_accuracy)

  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels})

ii=0
for i in range(Ntemp):
  av=0.0
  for j in range(samples_per_T):
        batch=(mnist.test.images[ii,:].reshape((1,lx*lx)),mnist.test.labels[ii,:].reshape((1,numberlabels)))     
        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1]})
        av=av+res 
        #print ii, res  
        ii=ii+1 
  av=av/samples_per_T
  print i,av   
       

for ii in range(Ntemp):
  batch=(mnist.test.images[ii*samples_per_T:ii*samples_per_T+samples_per_T,:].reshape(samples_per_T,lx*lx), mnist.test.labels[ii*samples_per_T,:ii*samples_per_T+samples_per_T].reshape((samples_per_T,numberlabels)) )
   train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1]}) 
  print ii, train_accuracy
#  for j in range(samples_per_T):
#  batch=(mnist.test.images[ii,:].reshape((1,lx*lx)),mnist.test.labels[ii,:].reshape((1,numberlabels)))
#        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1]})
#        av=av+res
        #print ii, res
#        ii=ii+1
#  av=av/samples_per_T
#  print i,av







