import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

# Eliminamos cualquier proceso que se esté ejecutando e iniciamos solo este
K.clear_session()

# Establecemos el path donde se encuentrar los datos de test y validacion
test_data_path = './data/dataset'
validation_data_path = './data/validation'


""" PARAMETROS """
# Numero de iteraciones que se realzan
epoch = 20

# Alto y ancho de las imagenes que se le introducen a la red
height, width = 100, 100

# Numero de imagenes se la red va a procesar en cada epoch
batch_size = 32

# Numero de veces que se va a analizar la informacion en cada epoch
steps = 1000

# Al finalizar el entrenamiento se realizara la validacion de datos y cada validacion se analizara 200 veces
validation_steps = 200

conv1_filters = 32
conv2_filters = 64

# Tamano de los filtros (kernel) que se van a utilizar durante las convoluciones
filter1_size = (3,3)
filter2_size = (2,2)

# Tamano del filtro (kernel) que se utilizara en el MaxPooling
pool_size = (2,2)

# Numero de clases o tipos de imagenes que se van a usar (bulbasaur, charmander, mewtwo, pikachu, squirtle)
classes = 5

# Valor del learning reate en el entrenamiento
learning_rate = 0.001


""" PRE PROCESAMIENTO DE LAS IMAGENES """
# Modificamos las propiedades de las imagenes de test y validacion
    # rescale = Se cambian los valores de la imagen en parametros mas pequeños
    # shear_range = Gira las imagenes para que la red aprenda a analizar objetos en distitos angulos
    # zoom_range = En algunas imagenes hace zoo para que la red aprenda a identificar objetos con solo una parte del mismo
    # horizontal_flip = Gira horizontalmente las imagenes para que la red aprenda a identificar objetos en otras perspectivas

test_gen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

val_gen = ImageDataGenerator(
    rescale = 1./255
)

image_testing = test_gen.flow_from_directory(
    test_data_path,
    target_size = (height,width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

image_validation = val_gen.flow_from_directory(
    validation_data_path,
    target_size = (height,width),
    batch_size = batch_size,
    class_mode = 'categorical'
)


""" Crear la CNN """
# Establecemos que la red es secuencial (conjunto de capas apiladas u ordenadas entre ellas)
cnn = Sequential()

# Creamos la primera capa de la red
cnn.add(Convolution2D(conv1_filters, filter1_size, padding = 'same', input_shape = (height,width,3), activation = 'relu'))

# Creamos una capa de MaxPooling (Despues de una capa de convolucion siempre va una de MaxPooling)
cnn.add(MaxPooling2D(pool_size = pool_size))

##
cnn.add(Convolution2D(conv2_filters, filter2_size, padding = 'same', activation = 'relu'))

##
cnn.add(MaxPooling2D(pool_size = pool_size))


# Despues de las capas de convolucion se agregan las capas de neuronas para hacer el analisis

# Aplanamos los datos salientes en una sola dimension
cnn.add(Flatten())

# Agregamos una capa "normal" con 256 neuronas todas conectanas a la red anterior (Dense) y con una funcion de activacion relu
cnn.add(Dense(256, activation = 'relu'))

# Dropout(0.5) apaga el 50% de las nueronas aletaroriamente en cada paso para que la red aprenda distinotos caminos y no solo uno
cnn.add(Dropout(0.5))

# Agregamos una capa "normal" con 256 neuronas todas conectanas a la red anterior (Dense) y con una funcion de activacion relu
cnn.add(Dense(classes, activation = 'softmax'))

cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr = learning_rate), metrics = ['accuracy'])

cnn.fit(image_testing, steps_per_epoch = steps, epochs = epoch, validation_data = image_validation, validation_steps = validation_steps)

dir = './model'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('./model/model.h5')
cnn.save_weights('./model/weights.h5')