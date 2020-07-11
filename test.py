import os
from utils import load_images

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict_

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load model into GPU / CPU
model_path = "./DenseDepth/nyu.h5"
model = load_model(model_path, custom_objects=custom_objects, compile=False)

def return_output(images_list):    
    # Input images
    inputs = load_images(images_list)
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict_(model, inputs)

    print('\nOutput ({0}) images of size {1}.'.format(outputs.shape[0], outputs.shape[1:]))

    return outputs