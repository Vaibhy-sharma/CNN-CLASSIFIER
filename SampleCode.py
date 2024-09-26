
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[2]:


model = Sequential()
model.add(Conv2D(5, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(5, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))
#model.load_weights('Initial_weight')


# In[3]:


import glob


# In[4]:


circles=glob.glob("C:/Users/Vik/Dtaset/Circle/*")


# In[5]:


Squares=glob.glob("C:/Users/Vik/Dtaset/Square/*")


# In[6]:


import scipy
import scipy.misc
import numpy as np


# In[7]:


circles=np.array([ scipy.misc.imread(c) for c in circles ])


# In[8]:


Squares=np.array([ scipy.misc.imread(c) for c in Squares ])


# In[9]:


circles=circles.reshape(126, 28, 28, 1)


# In[10]:


Squares=Squares.reshape(126, 28, 28, 1)


# In[11]:


train_x=np.concatenate((circles, Squares))


# In[12]:


k=np.zeros((252,2))


# In[13]:


k[:127,0]=1


# In[14]:


k[127:,1]=1


# In[15]:


y_train=k


# In[16]:


model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])


# In[18]:


model.fit(train_x, y_train, batch_size=32, epochs=3)


# In[28]:


circles=glob.glob("C:/Users/Vik/Dtaset/CNN/Test/Circle/*")


# In[29]:


Squares=glob.glob("C:/Users/Vik/Dtaset/CNN/Test/Square/*")


# In[30]:


circles=np.array([ scipy.misc.imread(c) for c in circles ])


# In[31]:


Squares=np.array([ scipy.misc.imread(c) for c in Squares ])


# In[34]:


circles=circles.reshape(circles.shape[0], 28, 28, 1)


# In[35]:


Squares=Squares.reshape(Squares.shape[0], 28, 28, 1)


# In[36]:


test_x=np.concatenate((circles, Squares))


# In[79]:


k=np.zeros((48,2))


# In[80]:


k[:24,0]=1


# In[81]:


k[24:,1]=1


# In[82]:


y_test=k


# In[89]:


model.evaluate(test_x, y_test)

