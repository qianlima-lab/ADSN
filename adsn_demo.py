import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.pyplot import savefig
import tensorflow.contrib.layers as tcl
np.random.seed(12345)

def transfer_labels(labels):
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = np.shape(labels)[0]
	for i in range(num_samples):
		new_label = np.argwhere(indexes == labels[i])[0][0]
		labels[i] = new_label
	return labels, num_classes

def open_file(path_data,dataset, info):
	data_x = []
	data_y = []
	count = 0
	for line in open(path_data):
		count = count + 1
		row = [[np.float32(x)] for x in line.strip().split(',')]
		label = np.int32(row[0])
		row = np.array(row[1:])
		row_shape = np.shape(row)
		row_mean = np.mean(row[0:])
		data_x.append(row-np.kron(np.ones((row_shape[0],row_shape[1])),row_mean))
		data_y.append(label[0])		
	return  data_x, data_y
	
def loading_ucr(index):
	dir_path = './UCR_TS_Archive_2015'
	list_dir = os.listdir(dir_path)
	dataset = list_dir[index]	
	train_data = dir_path+'/'+dataset+'/'+dataset+'_TRAIN'
	test_data = dir_path+'/'+dataset+'/'+dataset+'_TEST'
	train_x, train_y = open_file(train_data,dataset,'train')
	test_x, test_y   = open_file(test_data,dataset,'test')
	return train_x, train_y, test_x, test_y, dataset

dir_path = 'UCR_TS_Archive_2015'
list_dir = os.listdir(dir_path)
index = list_dir.index('ECGFiveDays')
train_x, train_y, test_x, test_y, dataset_name = loading_ucr(index=index)
nb_train, len_series =  np.shape(train_x)[0], np.shape(train_x)[1]
nb_test = np.shape(test_x)[0]
train_x = np.reshape(train_x,[nb_train,len_series])
test_x = np.reshape(test_x,[nb_test,len_series])
train_y, nb_class = transfer_labels(train_y)
train_y = np_utils.to_categorical(train_y, nb_class)
test_y, _ = transfer_labels(test_y)
test_y = np_utils.to_categorical(test_y, nb_class)

learning_rate=0.001
nb_epoch=200
dropout=0.75
lam=0.05
lambda_similar=0.01
iteration=3
batch_size=5
nb_shapelet=120
ratio=[0.7,0.8]
len_shapelet=[np.int(ratio[0]*len_series),np.int(ratio[1]*len_series)]
nb_slice = [len_series-i+1 for i in len_shapelet]

class D_conv_0(object):
	def __init__(self):
		self.name = "D_conv0"
	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()
			x = tf.reshape(tf.transpose(x, perm=[0, 2, 1]),shape=[-1,len_shapelet[0],1,1])
			d = tcl.conv2d(x, num_outputs=16, padding="SAME", kernel_size=[3,1], stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			d = tcl.conv2d(d, num_outputs=32, padding="SAME", kernel_size=[3,1], stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			d = tcl.max_pool2d(d,kernel_size=[len_shapelet[0],1],stride=1,padding="VALID")
			d = tcl.flatten(d)
			logit = tcl.fully_connected(d, 1, activation_fn=None)					
		return logit
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]
class D_conv_1(object):
	def __init__(self):
		self.name = "D_conv1"
	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()
			x = tf.reshape(tf.transpose(x, perm=[0, 2, 1]),shape=[-1,len_shapelet[1],1,1])
			d = tcl.conv2d(x, num_outputs=16, padding="SAME", kernel_size=[3,1], stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			d = tcl.conv2d(d, num_outputs=32, padding="SAME", kernel_size=[3,1], stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			#d = tcl.conv2d(d, num_outputs=64, padding="SAME", kernel_size=[3,1], stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			d = tcl.max_pool2d(d,kernel_size=[len_shapelet[1],1],stride=1,padding="VALID")
			d = tcl.flatten(d)
			logit = tcl.fully_connected(d, 1, activation_fn=None)					
		return logit
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

with tf.name_scope('Input_layer'):
	y=tf.placeholder(tf.float32,[None,nb_class])
	x_full_slice_0=tf.placeholder(tf.float32,[None,len_shapelet[0],nb_slice[0]])
	x_full_slice_1=tf.placeholder(tf.float32,[None,len_shapelet[1],nb_slice[1]])
	x_slice_true_0=tf.placeholder(tf.float32,[None,len_shapelet[0], nb_shapelet])
	x_slice_true_1=tf.placeholder(tf.float32,[None,len_shapelet[1], nb_shapelet])
	keep_prob=tf.placeholder(tf.float32)
	x_slice_reshape_0=tf.reshape(x_full_slice_0,shape=[-1,len_shapelet[0],1,nb_slice[0]])
	x_slice_reshape_1=tf.reshape(x_full_slice_1,shape=[-1,len_shapelet[1],1,nb_slice[1]])

with tf.name_scope('Generator'):
	with tf.name_scope('Conv_Shapelet0_layer0'):
		w_shapelet0_encoder1 = tf.Variable(tf.truncated_normal([3,1,nb_slice[0],nb_shapelet],stddev=0.001))
		b_shapelet0_encoder1 = tf.Variable(tf.constant(0.1, shape=[nb_shapelet]))
		shapelet0_encoder1=tf.nn.conv2d(x_slice_reshape_0,w_shapelet0_encoder1,strides=[1,1,1,1],padding='SAME')
		shapelet0_encoder1=tf.nn.bias_add(shapelet0_encoder1, b_shapelet0_encoder1)
	with tf.name_scope('Conv_Shapelet1_layer0'):
		w_shapelet1_encoder1 = tf.Variable(tf.truncated_normal([3,1,nb_slice[1],nb_shapelet],stddev=0.001))
		b_shapelet1_encoder1 = tf.Variable(tf.constant(0.1, shape=[nb_shapelet]))
		shapelet1_encoder1=tf.nn.conv2d(x_slice_reshape_1,w_shapelet1_encoder1,strides=[1,1,1,1],padding='SAME')
		shapelet1_encoder1=tf.nn.bias_add(shapelet1_encoder1, b_shapelet1_encoder1)
	with tf.name_scope('shapelet0'):
		x_slice_0=tf.reshape(x_full_slice_0,shape=[-1,len_shapelet[0],nb_slice[0],1])
		x_slice_0=tf.tile(x_slice_0,[1,1,1,nb_shapelet])
		shapelet0_tmp=tf.tile(tf.reshape(shapelet0_encoder1,shape=[-1,len_shapelet[0],nb_shapelet,1]),(1,1,1,nb_shapelet))
		similarity_0=tf.exp(-tf.reduce_sum(tf.square(tf.tile(shapelet0_encoder1,(1,1,nb_shapelet,1))-shapelet0_tmp),1))
		shapelet_0_tile=tf.tile(shapelet0_encoder1,[1,1,nb_slice[0],1])
		distance_0=tf.sqrt(tf.reduce_sum(tf.square(x_slice_0-shapelet_0_tile),1))
		shapelet_transform_0=tf.reduce_min(distance_0,1)
	with tf.name_scope('shapelet1'):
		x_slice_1=tf.reshape(x_full_slice_1,shape=[-1,len_shapelet[1],nb_slice[1],1])
		x_slice_1=tf.tile(x_slice_1,[1,1,1,nb_shapelet])
		shapelet1_tmp=tf.tile(tf.reshape(shapelet1_encoder1,shape=[-1,len_shapelet[1],nb_shapelet,1]),(1,1,1,nb_shapelet))
		similarity_1=tf.exp(-tf.reduce_sum(tf.square(tf.tile(shapelet1_encoder1,(1,1,nb_shapelet,1))-shapelet1_tmp),1))
		shapelet_1_tile=tf.tile(shapelet1_encoder1,[1,1,nb_slice[1],1])
		distance_1=tf.sqrt(tf.reduce_sum(tf.square(x_slice_1-shapelet_1_tile),1))
		shapelet_transform_1=tf.reduce_min(distance_1,1)

	shapelet_transform=tf.concat([shapelet_transform_0,shapelet_transform_1],1)
	fc1=shapelet_transform

	with tf.name_scope('Output'):
		with tf.name_scope('weights'):
			wout = tf.Variable(tf.truncated_normal([nb_shapelet*len(len_shapelet),nb_class]))
		with tf.name_scope('bias'):
			bout = tf.Variable(tf.constant(0.1, shape=[nb_class])) 
		fc1=tf.nn.dropout(fc1,keep_prob)
		out_tmp1=tf.add(tf.matmul(fc1,wout),bout)
		out_tmp2=tf.nn.softmax(out_tmp1)

with tf.name_scope('Discriminator'):
	discriminator_0=D_conv_0()
	discriminator_1=D_conv_1()
	shapelet0_fake=tf.reshape(shapelet0_encoder1,shape=[-1,len_shapelet[0],nb_shapelet])
	shapelet1_fake=tf.reshape(shapelet1_encoder1,shape=[-1,len_shapelet[1],nb_shapelet])
	D_real_0 = discriminator_0(x_slice_true_0)
	D_fake_0 = discriminator_0(shapelet0_fake, reuse = True)
	D_real_1 = discriminator_1(x_slice_true_1)
	D_fake_1 = discriminator_1(shapelet1_fake, reuse = True)
	
with tf.name_scope('Loss'):
	D_loss_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_0, labels=tf.ones_like(D_real_0))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_0, labels=tf.zeros_like(D_fake_0)))
	D_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_1, labels=tf.ones_like(D_real_1))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_1, labels=tf.zeros_like(D_fake_1)))
	G_loss_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_0, labels=tf.ones_like(D_fake_0)))
	G_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_1, labels=tf.ones_like(D_fake_1)))
	G_loss=lam*(G_loss_0+G_loss_1)
	similarity_loss=lambda_similar*(tf.norm(similarity_0)+tf.norm(similarity_1))
	cls_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_tmp1,labels=y))
	model_loss = cls_loss+G_loss+similarity_loss

with tf.name_scope('Train'):
	D_solver_0 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss_0, var_list=discriminator_0.vars)
	D_solver_1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss_1, var_list=discriminator_1.vars)
	G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model_loss, var_list=[var for var in tf.global_variables() if 'Generator' in var.name])

with tf.name_scope('Accuracy'):
	correct_pred=tf.equal(tf.argmax(out_tmp2,1),tf.argmax(y,1))
	accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

nb_batch_train = np.int32(np.floor(nb_train/batch_size))
if nb_train%batch_size==0:
	nb_total = nb_train
else:
	nb_total = np.int32((nb_batch_train+1)*batch_size)
delta = np.int32(nb_total - nb_train)

test_slice_full_0 = np.zeros((nb_test, len_shapelet[0], nb_slice[0]))
test_slice_full_1 = np.zeros((nb_test, len_shapelet[1], nb_slice[1]))
for m in range(nb_test):
	for k in range(nb_slice[0]):
		test_slice_full_0[m,:,k]=test_x[m,k:k+len_shapelet[0]]
for m in range(nb_test):
	for k in range(nb_slice[1]):
		test_slice_full_1[m,:,k]=test_x[m,k:k+len_shapelet[1]]

gpu_options = tf.GPUOptions(allow_growth=True)
test_accuracy_collect=[]
train_accuracy_collect=[]
train_loss_collect=[]
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(init)
	for i in range(nb_epoch):
		print 'Epoch:'+str(i)+'/'+str(nb_epoch)
		train_data_full = np.zeros((nb_total, len_series))
		train_label_full = np.zeros((nb_total, nb_class))
		L_train = [x_train for x_train in range(nb_train)]
		np.random.shuffle(L_train)
		for m in range(nb_train):
			train_data_full[m,:] = train_x[L_train[m],:]
			train_label_full[m,:] = train_y[L_train[m],:]
		for m in range(delta):
			train_data_full[nb_train+m,:] = train_data_full[m,:]
			train_label_full[nb_train+m,:] = train_label_full[m,:]
		train_slice_full_0 = np.zeros((nb_total, len_shapelet[0], nb_slice[0]))
		train_slice_full_1 = np.zeros((nb_total, len_shapelet[1], nb_slice[1]))
		for m in range(nb_total):
			for k in range(nb_slice[0]):
				train_slice_full_0[m,:,k]=train_data_full[m,k:k+len_shapelet[0]]
			for k in range(nb_slice[1]):
				train_slice_full_1[m,:,k]=train_data_full[m,k:k+len_shapelet[1]]
		train_accuracy_tmp=[]
		train_loss_tmp=[]
		for j in range(nb_total/batch_size):
			batch_x = train_data_full[j*batch_size:(j+1)*batch_size]
			batch_y = train_label_full[j*batch_size:(j+1)*batch_size]
			batch_slice_0 = train_slice_full_0[j*batch_size:(j+1)*batch_size]
			batch_slice_1 = train_slice_full_1[j*batch_size:(j+1)*batch_size]
			batch_slice_true_0 = np.zeros((batch_size, len_shapelet[0], nb_shapelet))
			batch_slice_true_1 = np.zeros((batch_size, len_shapelet[1], nb_shapelet))
			if nb_slice[0]>=nb_shapelet:
				L_true_0 = [x_true for x_true in range(nb_slice[0])]
				np.random.shuffle(L_true_0)
				for ind in range(nb_shapelet):
					batch_slice_true_0[:,:,ind]=batch_slice_0[:,:,L_true_0[ind]]
			else:
				L_true_0 = [x_true for x_true in range(nb_slice[0])]
				np.random.shuffle(L_true_0)
				base_0=np.int(np.floor(np.float(nb_shapelet)/nb_slice[0]))
				batch_slice_true_0[:,:,0:nb_slice[0]*base_0]=np.tile(batch_slice_0,(1,1,base_0))
				batch_slice_true_0[:,:,nb_slice[0]*base_0:nb_shapelet]= batch_slice_0[:,:,L_true_0[0:nb_shapelet-nb_slice[0]*base_0]]
			if nb_slice[1]>=nb_shapelet:
				L_true_1 = [x_true for x_true in range(nb_slice[1])]
				np.random.shuffle(L_true_1)
				for ind in range(nb_shapelet):
					batch_slice_true_1[:,:,ind]=batch_slice_1[:,:,L_true_1[ind]]
			else:
				L_true_1 = [x_true for x_true in range(nb_slice[1])]
				np.random.shuffle(L_true_1)
				base_1=np.int(np.floor(np.float(nb_shapelet)/nb_slice[1]))
				batch_slice_true_1[:,:,0:nb_slice[1]*base_1]=np.tile(batch_slice_1,(1,1,base_1))
				batch_slice_true_1[:,:,nb_slice[1]*base_1:nb_shapelet]= batch_slice_1[:,:,L_true_1[0:nb_shapelet-nb_slice[1]*base_1]]
			_,_,acc,model_los=sess.run([D_solver_0,D_solver_1,accuracy,model_loss],feed_dict={y:batch_y,x_full_slice_0:batch_slice_0,x_full_slice_1:batch_slice_1,x_slice_true_0:batch_slice_true_0,x_slice_true_1:batch_slice_true_1,keep_prob:dropout})
			train_accuracy_tmp.append(acc)
			train_loss_tmp.append(model_los)
			print("epoch: " + str(i) + ", batch: " + str(j) + ", accuracy= " + "{:.5f}".format(acc)+",loss= "+str(model_los))
			for ite in range(iteration):
				sess.run(G_solver,feed_dict={y:batch_y,x_full_slice_0:batch_slice_0,x_full_slice_1:batch_slice_1,x_slice_true_0:batch_slice_true_0,x_slice_true_1:batch_slice_true_1,keep_prob:dropout})
		train_accuracy_collect.append(np.mean(train_accuracy_tmp))
		train_loss_collect.append(np.mean(train_loss_tmp))
		test_accuracy_tmp=[]
		output_labels=[]
		if nb_test%batch_size==0:
			nb_batch_test = nb_test/batch_size
		else:
			nb_batch_test = (nb_test/batch_size)+1
		for j in range(nb_batch_test):
			prediction_acc,out_2=sess.run([correct_pred,out_tmp2],feed_dict={y:test_y[j*batch_size:(j+1)*batch_size],x_full_slice_0:test_slice_full_0[j*batch_size:(j+1)*batch_size],x_full_slice_1:test_slice_full_1[j*batch_size:(j+1)*batch_size],keep_prob: 1.0})
			test_accuracy_tmp=np.concatenate((test_accuracy_tmp,prediction_acc))
			output_labels=np.concatenate((output_labels,np.argmax(out_2,1)))
		test_accuracy=np.mean(test_accuracy_tmp[0:nb_test])
		test_accuracy_collect.append(test_accuracy)
		print("epoch: " + str(i) + ", testing accuracy= " + "{:.5f}".format(test_accuracy))
min_trainlos_acc=test_accuracy_collect[train_loss_collect.index(min(train_loss_collect))]
print("minloss accuracy:",min_trainlos_acc)
print('batch_size:',batch_size)
print('nb_shapelet:',nb_shapelet)
print('len_shapelet:',[ratio[0],ratio[1]])
print('dataset:',list_dir[index])
plt.figure(figsize=(9,4))
plt.plot(train_accuracy_collect,linewidth=0.5)
plt.plot(test_accuracy_collect,linewidth=0.5)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'], loc='upper left')
savefig('./plot/'+dataset_name+'_acc.pdf')
plt.show()
