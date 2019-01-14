import numpy as np
import cv2
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

import tkinter as tk
from tkinter import filedialog

# Metodo para abrir un filedialog y seleccionar una imagen a predecir
def load_data():

    root = tk.Tk()
    root.withdraw()

    #Abrimos el filedialog para buscar la imagen
    file_path = filedialog.askopenfilename()

    print(file_path)

    return file_path

longitud, altura = 100,100

# Importamos el archivo del modelo ya entrenado
modelo = './model/model.h5'
# Importamos el archivo con los valores de los pesos de la red
pesos = './model/weights.h5'
# Se asigna el modelo y pesos a usar
cnn = load_model(modelo)
cnn.load_weights(pesos)

def prediction(file):


    """
    path = cv2.imread(file)

    image = cv2.resize(path,(100,100))

    cv2.imwrite('data/validation/prueba.jpg', image)

    """
    
    x = load_img(file, target_size=(longitud,altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis = 0)

    arreglo = cnn.predict(x)

    resultado = arreglo[0]

    respuesta = np.argmax(resultado)

    if respuesta == 0:
        
        print('\n EN LA IMAGEN ESTA UN BULBASAUR')
    
    elif respuesta == 1:

        print('\n EN LA IMAGEN ESTA UN CHARMANDER')

    elif respuesta == 2:

        print('\n EN LA IMAGEN ESTA UN MEWTWO')

    elif respuesta == 3:

        print('\n EN LA IMAGEN ESTA UN PIKACHU')

    elif respuesta == 4:

        print('\n EN LA IMAGEN ESTA UN SQUIRTLE')

    return respuesta

respuesta = 'Y'

while(respuesta == 'Y'):

    archivo = load_data()

    prediction(archivo)

    print('\n\nRepetir programa? Y/N')
    respuesta = input()

