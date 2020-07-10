import os

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict

def return_output(model_path, images_list):
    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(model_path, custom_objects=custom_objects, compile=False)

    print('\nModel loaded ({0}).'.format(model_path))

    # Input images
    inputs = images_list
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)

    print('\nOutput ({0}) images of size {1}.'.format(outputs.shape[0], outputs.shape[1:]))

