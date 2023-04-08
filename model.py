import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split 

from utils import *

def create_model():
    model = Sequential() 
    model.add(Conv2D(24, (5, 5), strides = (2, 2), input_shape = (66, 200, 3), activation = 'elu'))
    model.add(Conv2D(36, (5, 5), strides = (2, 2), activation = 'elu'))
    model.add(Conv2D(48, (5, 5), strides = (2, 2), activation = 'elu'))
    model.add(Conv2D(64, (3, 3), activation = 'elu'))
    model.add(Conv2D(64, (3, 3), activation = 'elu'))
    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))  
    return model 

def train_model():
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1)
    model_checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)
    h = model.fit(batch_generator(X_train, y_train, 128, True),
                  steps_per_epoch = 300,
                  validation_data = batch_generator(X_val, y_val, 128, False),
                  validation_steps = 200,
                  epochs = 10,
                  verbose = True,
                  shuffle = True,
                  callbacks = [early_stopping, model_checkpoint])

if __name__ == '__main__':
    data_dir = "./Dataset"
    angle_correction = 0.15 
    num_of_bins = 25
    sample_threshold = 800
    data = csv_reader(data_dir)
    _, bins = np.histogram(data['steering'], num_of_bins)  
    data = truncate(data, bins, num_of_bins, sample_threshold)
    image_paths, steering_angles = unpack(data, data_dir + '/IMG', angle_correction) 
    X_train, X_val, y_train, y_val = train_test_split(image_paths, steering_angles, test_size = 0.2, random_state = 33)  
    
    assert(X_train.shape[0] == y_train.shape[0]), "ERROR: The number of training images is not equal to the number of training labels"
    assert(X_val.shape[0] == y_val.shape[0]), "ERROR: The number of validation images is not equal to the number of validation labels"

    model = create_model()
    model.compile(Adam(learning_rate = 1e-4), loss = 'mse')
    print(model.summary())
    train_model()
    model.save('model.h5')
