import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
from PIL import Image

# 使用encode-decode网络结构不成功，将decode改为UNET训练成功，注意归一化使用问题
N = 70
H = 128
W = 128
c = 1
gam = 100
batch_size = 1
EPSILON = 1e-10

def deconv2d(inputs,f_size,out_shape,strides):
    fiters = tf.get_variable('fiter', shape=[f_size ,f_size ,out_shape ,inputs.shape[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable('b', shape=[out_shape], initializer=tf.constant_initializer([0.]))
    w = int(inputs.shape[1])
    h =int(inputs.shape[2])
    return tf.nn.conv2d_transpose(inputs,fiters,[batch_size,strides*w, strides *h,out_shape],[1,strides,strides,1]) + b

def conv2d(inputs,shape,strides):
    fiters = tf.get_variable('fiter',shape = shape,initializer=tf.random_normal_initializer(stddev = 0.02))
    b = tf.get_variable('b',shape = [shape[-1]],initializer= tf.constant_initializer ([0.]) )
    return tf.nn.conv2d(inputs,fiters,strides,'SAME') + b

def batch_norm(x):
    # return x
    mu,var = tf.nn.moments(x,axes = [1,2],keep_dims= True)
    gamma = tf.get_variable('gamma',shape = [mu.shape[-1]],initializer=tf.constant_initializer ([1.0])  )
    beta = tf.get_variable('beta',shape = [mu.shape[-1]],initializer= tf.constant_initializer ([0.]) )
    return gamma * (( x - mu ) / tf.sqrt ( var + 1e-14) ) + beta

def leak_relu(x, s = 0.2):
    return tf.maximum(x,s*x)

def fully_connect(x,out_shape):
    w = tf.get_variable('w',shape = [x.shape[-1],out_shape ],initializer=tf.random_normal_initializer(stddev = 0.02))
    b = tf.get_variable('b',shape = [out_shape],initializer= tf.constant_initializer ([0.]) )
    return tf.matmul(x,w) + b

def mapping(x):
    max = np.max(x)
    min = np.min(x)
    return (x-min)*255.0/(max - min)

#encode-decode
class Generator():
    def __init__(self,name):
        self.name = name
    def __call__(self,x):
        with tf.variable_scope (name_or_scope= self.name,reuse = False ):
            with tf.variable_scope(name_or_scope= 'linear'):
                input0 = leak_relu(batch_norm(conv2d(x,[4,4,c,64],[1,2,2,1])) )
            with tf.variable_scope(name_or_scope='conv1' ):
                input1 = leak_relu(batch_norm(conv2d(input0,[4,4,64,128],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv2'):
                input2 = leak_relu(batch_norm(conv2d(input1,[4,4,128,256],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv3'):
                input3 = leak_relu(batch_norm(conv2d(input2,[4,4,256,512],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv4'):
                input4 = leak_relu(batch_norm(conv2d(input3,[4,4,512,512],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv5'):
                input5 = leak_relu(batch_norm(conv2d(input4,[4,4,512,512],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv6'):
                input6 = leak_relu(batch_norm(conv2d(input5,[4,4,512,512],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv7'):
                input7 = leak_relu(batch_norm(conv2d(input6,[4,4,512,512],[1,2,2,1])) )

            with tf.variable_scope (name_or_scope='dconv1'):
                deconv = batch_norm(deconv2d(input7, 4,512,2))
                input8 = tf.nn.relu(tf.nn.dropout(deconv ,keep_prob = 0.5 ))
            with tf.variable_scope(name_or_scope='deconv2'):
                deconv = batch_norm(deconv2d(input8, 4,512,2))
                input9 = tf.nn.relu(tf.nn.dropout(deconv, keep_prob=0.5))
            with tf.variable_scope(name_or_scope='deconv3'):
                deconv = batch_norm(deconv2d(input9, 4,512,2))
                input10 = tf.nn.relu(tf.nn.dropout(deconv, keep_prob=0.5))
            with tf.variable_scope(name_or_scope='deconv4'):
                input11 = tf.nn.relu(batch_norm(deconv2d(input10, 4,512,2)))
            with tf.variable_scope(name_or_scope='deconv5'):
                input12 = tf.nn.relu(batch_norm(deconv2d(input11, 4,512,2)))
            with tf.variable_scope(name_or_scope='deconv6'):
                input13 = tf.nn.relu(batch_norm(deconv2d(input12,4,256,2) ) )
            with tf.variable_scope(name_or_scope='deconv7'):
                input14 = tf.nn.relu(batch_norm(deconv2d(input13,4,128,2)) )
            with tf.variable_scope(name_or_scope= 'deconv8'):
                input15 = tf.nn.relu(batch_norm(deconv2d(input14,4,64,2)) )
            with tf.variable_scope(name_or_scope='deconv9'):
                return tf.nn.tanh(conv2d(input15,[4,4,64,3],[1,1,1,1]))
    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

#encode-UNet
class Generator1():
    def __init__(self,name):
        self.name = name
    def __call__(self,x):
        with tf.variable_scope (name_or_scope= self.name,reuse = tf.AUTO_REUSE ):
            with tf.variable_scope(name_or_scope= 'conv0'):
                input0 = leak_relu(batch_norm(conv2d(x,[4,4,c,64],[1,2,2,1])) )
            with tf.variable_scope(name_or_scope='conv1' ):
                input1 = leak_relu(batch_norm(conv2d(input0,[4,4,64,128],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv2'):
                input2 = leak_relu(batch_norm(conv2d(input1,[4,4,128,256],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv3'):
                input3 = leak_relu(batch_norm(conv2d(input2,[4,4,256,256],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv4'):
                input4 = leak_relu(batch_norm(conv2d(input3,[4,4,256,256],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv5'):
                input5 = leak_relu(batch_norm(conv2d(input4,[4,4,256,256],[1,2,2,1])) )
            with tf.variable_scope (name_or_scope= 'conv6'):
                input6 = leak_relu(batch_norm(conv2d(input5,[4,4,256,256],[1,2,2,1])) )
            #with tf.variable_scope (name_or_scope= 'conv7'):
                #input7 = leak_relu(batch_norm(conv2d(input6,[4,4,256,256],[1,2,2,1])) )
            #with tf.variable_scope (name_or_scope='dconv1'):
                #deconv = batch_norm(deconv2d(input7,4,256,2))
                #input8 = tf.concat([tf.nn.relu(tf.nn.dropout(deconv ,keep_prob = 0.5 )),input6],axis = 3)
            with tf.variable_scope(name_or_scope='deconv2'):
                deconv = batch_norm(deconv2d(input6, 4,256,2))
                input9 = tf.concat([tf.nn.relu(tf.nn.dropout(deconv, keep_prob=0.5)),input5],axis = 3)
            with tf.variable_scope(name_or_scope='deconv3'):
                deconv = batch_norm(deconv2d(input9, 4,256,2))
                input10 = tf.concat([tf.nn.relu(tf.nn.dropout(deconv, keep_prob=0.5)),input4],axis = 3)
            with tf.variable_scope(name_or_scope='deconv4'):
                input11 = tf.concat([tf.nn.relu(batch_norm(deconv2d(input10, 4,256,2))) ,input3],axis = 3)
            with tf.variable_scope(name_or_scope='deconv5'):
                input12 = tf.concat([tf.nn.relu(batch_norm(deconv2d(input11, 4,256,2))) ,input2],axis = 3)
            with tf.variable_scope(name_or_scope='deconv6'):
                input13 = tf.concat([tf.nn.relu(batch_norm(deconv2d(input12,4,256,2) ) ) , input1],axis = 3)
            with tf.variable_scope(name_or_scope='deconv7'):
                input14 = tf.concat([tf.nn.relu(batch_norm(deconv2d(input13,4,256,2)) ), input0],axis = 3)
            with tf.variable_scope(name_or_scope= 'deconv8'):
                input15 = tf.nn.relu(batch_norm(deconv2d(input14,4,128,2)) )
            with tf.variable_scope(name_or_scope='deconv9'):
                return tf.nn.tanh(conv2d(input15,[4,4,128,1],[1,1,1,1]))
    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class Discriminator:
    def __init__(self,name):
        self.name = name

    def __call__(self,x,y,reuse):
        input = tf.random_crop(tf.concat(axis=3, values=(x, y)), [batch_size, N, N, 2*c])
        with tf.variable_scope(name_or_scope=self.name,reuse = reuse):
            with tf.variable_scope(name_or_scope= 'deconv1'):
                inputs0 = leak_relu(conv2d(input,[4,4,2*c,64],[1,2,2,1]) )
            with tf.variable_scope (name_or_scope= 'deconv2'):
                inputs1 = leak_relu(batch_norm(conv2d(inputs0,[4,4,64,128],[1,2,2,1]) ) )
            with tf.variable_scope (name_or_scope= 'deconv3'):
                inputs2 = leak_relu(batch_norm(conv2d(inputs1,[4,4,128,256],[1,2,2,1]) ) )
            with tf.variable_scope (name_or_scope= 'deconv4'):
                inputs3 = leak_relu(batch_norm(conv2d(inputs2,[4,4,256,512],[1,2,2,1]) ) )
            return fully_connect(tf.layers.flatten(inputs3),1 )

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class cGAN:
    def __init__(self):
        self.x = tf.placeholder('float',shape = [batch_size ,H,W,c])
        self.y = tf.placeholder('float',shape = [batch_size ,H,W,c])
        G = Generator('generator')
        G1 = Generator1('genenrator1')
        D = Discriminator('discriminator')
        self.feal_y = G1(self.x)
        self.feal_logit = tf.nn.sigmoid(D(self.x,self.feal_y,False))
        self.real_logit = tf.nn.sigmoid(D(self.x,self.y,True))
        self.dloss = -(tf.reduce_mean(tf.log(self.real_logit + EPSILON)) + tf.reduce_mean(tf.log(1 - self.feal_logit + EPSILON )))
        self.L1 = tf.reduce_mean(tf.abs(self.y - self.feal_y))
        self.gloss = -(tf.reduce_mean(tf.log(self.feal_logit + EPSILON ))) + gam * self.L1

        self.opg = tf.train.AdamOptimizer(learning_rate=2e-4) .minimize(self.gloss ,var_list=G1.var)
        self.opd = tf.train.AdamOptimizer(learning_rate= 2e-4) .minimize(self.dloss,var_list = D.var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self):
        saver = tf.train.Saver ()
        traindata_name = os.listdir('./cityscapes/train')
        for i in os.listdir('./cityscapes/train') :
            traindata_name.append(i)
        # trainA = os.listdir('./single-dataset/0')
        # trainB = os.listdir('./single-dataset/2')
        epochs_nums = 200
        saver.restore(self.sess, "E:/pythonproject/TensorFlow/para/cGANmodel.ckpt")
        for epochs in range(epochs_nums ):
            for i in range(traindata_name.__len__() ):
                X = np.array(Image.open('./cityscapes/train/' + traindata_name[i]))[np.newaxis,:,256:,:] / 127.5 - 1.0
                Y = np.array(Image.open('./cityscapes/train/' + traindata_name[i]))[np.newaxis,:,0:256,:] / 127.5 - 1.0
                # X = np.array(Image.open('./single-dataset/0/' + trainA[i]))[np.newaxis, :, :, np.newaxis] / 127.5 - 1.0
                # Y = np.array(Image.open('./single-dataset/2/' + trainB[i]))[np.newaxis, :, :, np.newaxis] / 127.5 - 1.0
                gloss = self.sess.run(self.gloss,feed_dict= {self.x: X,self.y: Y})
                dloss = self.sess.run(self.dloss,feed_dict={ self.x: X,self.y: Y})
                self.sess.run(self.opg ,feed_dict={ self.x: X,self.y: Y})
                self.sess.run(self.opd,feed_dict={ self.x: X,self.y: Y})
                if i % 10 == 0:
                    print('epoch:%d, step: %d,gloss: %f,dloss: %f' % (epochs,i,gloss,dloss))
                    image = self.sess.run(self.feal_y,feed_dict={ self.x: X,self.y: Y})
                    #image = np.concatenate((mapping(Y[0, :, :, :]), mapping(image[0, :, :, :])), axis=1)
                    image = np.concatenate((mapping(Y[0, :, :, 0]), mapping(image[0, :, :, 0])), axis=1)
                    if c == 1:
                        Image.fromarray(np.uint8(image)).save("./cGANresult//" + str(epochs) + "_" + str(i) + ".jpg")
                    else:
                        Image.fromarray(np.uint8(image)).save("./cGANresult//" + str(epochs) + "_" + str(i) + ".jpg")
            saver.save(self.sess, "./para//cGANmodel.ckpt")
if __name__ == "__main__":
    gan = cGAN()
    gan()
















