import tensorflow as tf
import numpy as np
import pandas as pd
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt	

################################################
# This program uses a 5-layers CNN to conduct 
# Radar images extrapolation, which is crucial  
# for nowcasting (next-hour rainfall prediction)
################################################


def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1,seed=12)
    initial = tf.get_variable("W", shape,
           initializer=tf.contrib.layers.xavier_initializer())
    return initial

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    
#input 4*280*280
x = tf.placeholder(tf.float32, [None, 280,280,4])
keep_prob = tf.placeholder(tf.float32)
x_drop = tf.nn.dropout(x, keep_prob)
y_true = tf.placeholder(tf.float32, [None,1])
with tf.variable_scope("conv1"):
    W_conv1 = weight_variable([9,9,4,12])
    b_conv1 = bias_variable([12])
    # c1 12*272*272
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1,strides = [1,1,1,1], padding = 'VALID') + b_conv1) 
# s1 12*136*136
h_pool1 = max_pool_2x2(h_conv1)
#h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)
# c2 32*128*128
with tf.variable_scope("conv2"):
    W_conv2 = weight_variable([9,9,12,32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides = [1,1,1,1], padding = 'VALID') + b_conv2) 
# s2 32*64*64
h_pool2 = max_pool_2x2(h_conv2)
#h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)
# c3 32*56*56
with tf.variable_scope("conv3"):
    W_conv3 = weight_variable([9,9,32,32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3,strides = [1,1,1,1], padding = 'VALID') + b_conv2) 
# s3 32*28*28
h_pool3 = max_pool_2x2(h_conv3)
# c4 32*20*20
with tf.variable_scope("conv4"):
    W_conv4 = weight_variable([9,9,32,32])
    b_conv4 = bias_variable([32])
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4,strides = [1,1,1,1], padding = 'VALID') + b_conv2) 
# s4 32*10*10
h_pool4 = max_pool_2x2(h_conv4)
# c5 32*4*4
with tf.variable_scope("conv5"):
    W_conv5 = weight_variable([7,7,32,32])
    b_conv5 = bias_variable([32])
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4, W_conv5,strides = [1,1,1,1], padding = 'VALID') + b_conv2) 
# flat
h_flat = tf.reshape(h_conv5, [-1,4*4*32])
with tf.variable_scope("flat_V"):
    WV = weight_variable([512, 41])
    BV = bias_variable([41])
    VPV=tf.transpose(tf.nn.softmax(tf.matmul(h_flat, WV) + BV))    #############
with tf.variable_scope("flat_H"):
    WH = weight_variable([512, 41])
    BH = bias_variable([41])
    HPV=tf.nn.softmax(tf.matmul(h_flat, WH) + BH)

y0 = tf.placeholder(tf.float32, [None, 280,280,1])
y_true = tf.placeholder(tf.float32, [None, 240,240,1])
DC1=tf.nn.conv2d(y0, tf.expand_dims(tf.expand_dims(VPV, 1),1),strides = [1,1,1,1], padding = 'VALID')
y_pred=tf.nn.conv2d(DC1, tf.expand_dims(tf.expand_dims(HPV, -1),-1),strides = [1,1,1,1], padding = 'VALID')
cost = tf.reduce_sum(tf.pow(y_pred-y_true, 2))
# learning rate decay
#lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
#step = tf.placeholder(tf.int32)    # fed by i

# training methods:
#train_step = tf.train.AdadeltaOptimizer(1.5e-3).minimize(cost) # result has "shadows"
train_step = tf.train.AdamOptimizer(1.5e-5).minimize(cost)

# start the model:
trn_img=np.zeros([1,280,280,4])
trn_img[0,:,:,0]= np.load('201805031818_gray.npy').astype('float32')   
trn_img[0,:,:,1]= np.load('201805031824_gray.npy').astype('float32')    
trn_img[0,:,:,2]= np.load('201805031830_gray.npy').astype('float32')    
trn_img[0,:,:,3]= np.load('201805031836_gray.npy').astype('float32')
trn_lst=np.zeros([1,280,280,1])
trn_lst[0,:,:,0]=np.load('201805031836_gray.npy').astype('float32') 
y1=np.load('201805031842_gray.npy').astype('float32')    
Y=np.zeros([1,240,240,1])
Y[0,:,:,0]=y1[20:260,20:260]
#plt.imshow(trn_img[0,:,:,3],cmap='gray')    
#plt.savefig('201805031836') 
#exit()  
iter_list = []
train_loss_list = []
val_loss_list = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(50):
        _,loss_value= sess.run([train_step,cost],feed_dict={x:trn_img,y0:trn_lst,y_true:Y})
        print(loss_value)
    test_result = sess.run([y_pred],feed_dict = {x:trn_img,y0:trn_lst,y_true:Y})
import os
os.system('del CNN_out.png')
plt.imshow(np.array(test_result)[0,0,:,:,0],cmap='gray')    
plt.savefig('CNN_out')    
exit() 

tst_pic['value'] = test_result[0]
print(np.mean(test_result[0]))
tst_pic.to_csv(data_folder + 'result_cnn.csv',index = False)
valid_curve = pd.DataFrame({'iter':iter_list,'train_loss':train_loss_list,'val_loss':val_loss_list})
valid_curve.to_csv(data_folder + 'result_cnn_curve.csv',index = False)

