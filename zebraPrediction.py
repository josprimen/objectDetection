from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#cargamos el modelo keras que creamos con los pesos pre-entrenados
model = load_model('model.h5')

#creamos una función que englobe toda la funcionalidad de carga y tratado de imagen
def load_image_pixels(filename, shape):
    # cargar la imagen para guardar su formato
    image = load_img(filename)
    width, height = image.size

    #cargar la ilmagen con el tamaño requerido
    image = load_img(filename, target_size=shape)
    #convertir a array numpy
    image = img_to_array(image)
    #escalar pixels en valores [0,1]
    image = image.astype('float32')
    image /= 255.0

    # agregue una dimensión para que tengamos una muestra
    image = expand_dims(image, 0)
    return image, width, height

#Definimos el tamaño objetivo y la foto
input_w, input_h = 416, 416
photo_filename = 'zebra.jpg'
#cargamos y preparamos la imagen usando nuestra función
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

#Cargamos la foto en el modelo Keras y hacemos una predicción
yhat = model.predict(image)
# resume la forma de la lista de matrices
print([a.shape for a in yhat])