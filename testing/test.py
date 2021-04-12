import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

######################################

AUTOTUNE = tf.data.experimental.AUTOTUNE
desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

#test check
print('starting.....')

# The 1./255 is to convert from uint8 to float32 in range [0,1].
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 126
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(1011+250/BATCH_SIZE)

train_data_gen = train_image_generator.flow_from_directory(directory=str('../images/train'),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(directory=str('../images/test'),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode='binary')

validation_data_gen = test_image_generator.flow_from_directory(directory=str('../images/validation'),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode='binary')

#print(str(desktop+'/image_sets'))



def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title('hand')
      plt.axis('off')

image_batch, label_batch = next(train_data_gen)
#show_batch(image_batch, label_batch)
#plt.show()

# image shape: batch x height x width x color channel
#print(image_batch.shape)

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
# Let's take a look at the base model architecture


### FEATURE EXTRACTION ###

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.00001


model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.summary()
#len(model.trainable_variables)



initial_epochs = 1
validation_steps= 40

#loss0,accuracy0 = model.evaluate(validation_data_gen, steps = validation_steps)

history = model.fit(train_data_gen,
                    epochs=initial_epochs,
                    validation_data=validation_data_gen) #steps_per_epoch=STEPS_PER_EPOCH)


# Save the entire model as a SavedModel.
model.save('saved_model/my_model')

'''
# Export the model to a SavedModel
model.save('path_to_saved_model', save_format='tf')

# Recreate the exact same model
new_model = tf.keras.models.load_model('path_to_saved_model')
print(new_model)
'''

#tf.saved_model.save(model, "/tmp/custModel/1/")
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
'''

###########################################
# Initialize all variables


'''
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
saver = tf.compat.v1.train.Saver()
saver.save(sess,'./tensorflowModel.ckpt')
tf.io.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)

freeze_graph.freeze_graph('tensorflowModel.pbtxt', "", False, 
                      './tensorflowModel.ckpt', "output/softmax",
                        "save/restore_all", "save/Const:0",
                        'frozentensorflowModel.pb', True, ""  
                      )
'''