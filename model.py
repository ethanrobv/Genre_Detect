import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp

# need to define custom layers in order to override get_config()
class CustomDenseVariational(tfp.layers.DenseVariational):
    def __init__(self, units, make_posterior_fn, make_prior_fn, kl_weight=None, **kwargs):
        self.units = units
        self.make_posterior_fn = make_posterior_fn
        self.make_prior_fn = make_prior_fn
        self.kl_weight = kl_weight
        super(CustomDenseVariational, self).__init__(units, make_posterior_fn, make_prior_fn, kl_weight, **kwargs)

    def get_config(self):
        config = super(CustomDenseVariational, self).get_config()
        config.update({'units': self.units,
                       'make_posterior_fn': self.make_posterior_fn,
                       'make_prior_fn': self.make_prior_fn,
                       'kl_weight': self.kl_weight})
        return config

#-------------Data Prep-----------------#

#load data from data.csv
data=np.loadtxt(fname='./data.csv',delimiter=',',dtype=float,skiprows=1)

#remove first column
data = np.delete(data,0,1)

#shuffle
np.random.shuffle(data)

#split into X and y
y,X=np.split(data,[1],1)
y=[y[i][0] for i in range(len(y))] #during the split y is made into a 2d array, this makes it 1d

#make train and test sets
half = int(len(X)/2)
split = half + (int)(half/2) #split data at 75% for training and 25% for testing
xTrain = X[:split]
yTrain = y[:split]
xTest = X[split:]
yTest = y[split:]

#make sure all sets are correct type
xTrain = np.asarray(xTrain)
yTrain = np.asarray(yTrain)
xTest = np.asarray(xTest)
yTest = np.asarray(yTest)


yTrain = keras.utils.to_categorical(yTrain) #if its hiphop (0) itll be [1, 0] if its rock (1) itll be [0, 1]
yTest = keras.utils.to_categorical(yTest)


#----------------BNN----------------#

def run_model(model, x_train, y_train, x_test, y_test, num_epochs):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy']
    )

    print("Start training the model...")
    model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))
    print("Model training finished")

    # use the model to make predictions on the test data
    predictions = model.predict(x_test)

    # use the argmax function to obtain the predicted class labels
    predicted_classes = np.argmax(predictions, axis=-1)



    print(predicted_classes)

    for i in range(len(x_test)):
        if predicted_classes[i] == y_test[i][1]:
            print("works")
        else:
            print("dont works")
    

def create_model_inputs():
    inputs = {}
    for feature in range(len(xTrain[0])):
        inputs[feature] = layers.Input(shape=(1,), dtype=tf.float32)
    return inputs

def prior_437(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior_437(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model 


def create_model():

    inputs = keras.Input(shape=(len(xTrain[0]),))

    norm = layers.BatchNormalization()(inputs)

    dense = keras.layers.Dense(256)(norm)
    dense = keras.layers.LeakyReLU(alpha=0.3)(dense)

    dense = keras.layers.Dense(128)(dense)
    dense = keras.layers.LeakyReLU(alpha=0.3)(dense)

    dense = CustomDenseVariational(units=15, make_prior_fn=prior_437, make_posterior_fn=posterior_437, kl_weight=1 / len(xTrain[0]), activation="sigmoid")(dense)


    outputs = layers.Dense(units=2, activation="softmax")(dense)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    model = create_model()
    #model.load_weights('model_weights.h5')
    model.summary()
    run_model(model, xTrain, yTrain, xTest, yTest, 200)
    model.save_weights('./model_weights.h5')