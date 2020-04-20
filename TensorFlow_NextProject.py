
import csv
import tensorflow as tf
from tensorflow import keras
import numpy
from tensorflow.keras.preprocessing.text import Tokenizer

import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
import os.path


WordCount = 10000
Examples = 480000


BatchSize = 128
max_epochs = 5
#input()
csv_path = r"C:\Users\reube\source\repos\TensorFlow_Projects\Datasets\rotten_tomatoes_reviewsEdited.csv"

SaveAt = r"C:\Users\reube\source\repos\TensorFlow_Projects\Models\Rotten Tomatoes\Large Set" + "\\"
print(SaveAt)






AmountCounted = 0

JustReviews = []
print("opening file" + csv_path)
with open(csv_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader) #Skip header
    for lines in csv_reader:
      JustReviews.append(lines[1])
      AmountCounted = AmountCounted +1
      if(AmountCounted % 1000 ==0):
       
        print("Counted: " + str(AmountCounted) + " (%.2f" % (100*AmountCounted/Examples) + "%)")
print("Got Reviews")



#input()
print("Making Word index")
tokenizer = Tokenizer(num_words=WordCount)
tokenizer.fit_on_texts(JustReviews)
word_index = tokenizer.word_index
#print(word_index)
print("Finsihed Making Word Index")
print("Converting Reviews to a matrix")
MatrixedReviews = tokenizer.texts_to_matrix(JustReviews)
#print(MatrixedReviews)

AmountFCounted = 0


def create_dataset(xs, ys, n_classes=2):
  #ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
     .shuffle(len(ys)) \
    .batch(BatchSize)



print("Getting Freshness Ratings")
Freshnessratings = []
with open(csv_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)#Skip header
    for lines in csv_reader:
      Freshnessratings.append(float(lines[0]))
      AmountFCounted = AmountFCounted +1
      if(AmountFCounted % 1000 ==0):
        
        print("Counted: " + str(AmountFCounted) + " Freshnesses ( %.2f" % (100*AmountFCounted/Examples) + "%)")
print("Got Ratings")

X_train = MatrixedReviews
Y_train = Freshnessratings

#------------------------------ Sequence generator to manage RAM
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
class DataSequence(Sequence):   
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
    def __len__(self):
   #     print( "Amount of Batches:" + str(math.ceil(len(self.x) / self.batch_size)))
        return math.ceil(len(self.x) / self.batch_size)
    def __getitem__(self, idx):
       #   print("Getting Batch index"+ str(idx))
          batch_x = self.x[idx * self.batch_size:(idx + 1) *self.batch_size]
          
          batch_y = self.y[idx * self.batch_size:(idx + 1) *self.batch_size]
          
  

          return numpy.array(batch_x), numpy.array(batch_y)
                











#----------------------- Neural Network

#https://visualstudiomagazine.com/articles/2018/08/30/neural-binary-classification-keras.aspx

train_x =  X_train[:int(0.75*Examples)]
train_y = Y_train[:int(0.75*Examples)]
test_x= X_train[int(0.75*Examples):]
test_y =  Y_train[int(0.75*Examples):]

TestSeq = DataSequence(test_x,test_y,BatchSize)
TrainSeq = DataSequence(train_x,train_y,BatchSize)


#def on_epoch_end(self, epoch, logs={}):
#    if epoch % self.n == 0:
#      curr_loss =logs.get('loss')
#      curr_acc = logs.get('acc') * 100
#      print("epoch = %4d loss = %0.6f acc = %0.2f%%" % \
#        (epoch, curr_loss, curr_acc))




my_init = keras.initializers.glorot_uniform(seed=1)
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=WordCount, input_dim=WordCount,
  activation='tanh', kernel_initializer=my_init)) 
model.add(keras.layers.Dense(units=WordCount, activation='tanh',
  kernel_initializer=my_init)) 
model.add(keras.layers.Dense(units=1, activation='sigmoid',
  kernel_initializer=my_init))

simple_sgd = keras.optimizers.SGD(lr=0.01)  
model.compile(loss='binary_crossentropy',
  optimizer=simple_sgd, metrics=['accuracy'])  



class MyLogger(keras.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs
    print(n)


my_logger = MyLogger(n=1)

h = model.fit(TrainSeq,
epochs=max_epochs, verbose=1,)#, callbacks=[my_logger]) #, batch_size=BatchSize,  steps_per_epoch=10




numpy.set_printoptions(precision=4, suppress=True)
eval_results = model.evaluate(TestSeq, verbose=1) #cant pass as a nparray Also should be test sets?  ,batch_size=BatchSize
print("\nLoss, accuracy on test data: ")
print("%0.4f %0.2f%%" % (eval_results[0], \
  eval_results[1]*100))

#input()

#-----------------------------------------

#Traindataset =  create_dataset(X_train[:int(0.75*Examples)], Y_train[:int(0.75*Examples)])
#Testdataset =  create_dataset(X_train[int(0.75*Examples):], Y_train[int(0.75*Examples):])
#model = keras.Sequential([
#   # keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(WordCount,1)),
#    keras.layers.Dense(units=WordCount, activation='relu'),
#    #keras.layers.Dense(units=192, activation='relu'),
#    keras.layers.Dense(units=128, activation='relu'),
#    keras.layers.Dense(units=1, activation='softmax')
#])



#model.compile(optimizer='adam', 
#              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])


#history = model.fit(
#   Traindataset.repeat(), 
#    epochs=10, 
#    steps_per_epoch=500,
#    validation_data=Testdataset.repeat(), 
#    validation_steps=2
#)
 #----------------------------------------


ModelPath = SaveAt + "model.json"
WeightPath = SaveAt + "model.h5"


 # serialize model to JSON
model_json = model.to_json()
with open(ModelPath, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(WeightPath)
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

input()