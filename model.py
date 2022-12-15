import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp

#-------------Data Prep-----------------#

#load data from data.csv
data=np.loadtxt(fname='data.csv',delimiter=',',dtype=float,skiprows=1)

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

#print(xTrain[0])
#print(yTrain)

yTrain = keras.utils.to_categorical(yTrain) #if its hiphop (0) itll be [1, 0] if its rock (1) itll be [0, 1]
yTest = keras.utils.to_categorical(yTest)

#print(yTrain)

#print(yTrain)
#print(len(yTrain))
#print(len(xTrain))

#----------------BNN----------------#

def run_model(model, x_train, y_train, x_test, y_test, num_epochs):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy']
    )

    print("Start training the model...")
    model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))
    print("Model training finished")

    #print("Evaluation model performance...")
    #_, rmse = model.evaluate((x_train, y_train), verbose=0)
    #print(f"Train RMSE: {round(rmse, 3)}")

    # use the model to make predictions on the test data
    predictions = model.predict(x_test)

    # use the argmax function to obtain the predicted class labels
    predicted_classes = np.argmax(predictions, axis=-1)


    print(predicted_classes)

    for i in range(len(x_test)):
        if predicted_classes[i] == y_test[i][0]:
            print("works")
        else:
            print("dont works")
    

def create_model_inputs():
    inputs = {}
    for feature in range(len(xTrain[0])):
        inputs[feature] = layers.Input(shape=(1,), dtype=tf.float32)
    return inputs

def prior(kernel_size, bias_size, dtype=None):
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

def posterior(kernel_size, bias_size, dtype=None):
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


def create_model(train_size):

    inputs = keras.Input(shape=(int(len(xTrain[0])),))
    #features = keras.layers.concatenate(list(inputs.values()))
    #features = layers.BatchNormalization()(inputs)

    dense = tfp.layers.DenseVariational(units=32, make_prior_fn=prior,
        make_posterior_fn=posterior, kl_weight=1 / train_size, activation="relu")(inputs)

    dense = tfp.layers.DenseVariational(units=20, make_prior_fn=prior,
        make_posterior_fn=posterior, kl_weight=1 / train_size, activation="relu")(dense)

    #for units in hidden_units:
    #    features = tfp.layers.DenseVariational(units=units, make_prior_fn=prior,
    #    make_posterior_fn=posterior, kl_weight=1 / train_size, activation="sigmoid")(features)

    outputs = layers.Dense(units=2)(dense)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_model(len(xTrain))
model.summary()
run_model(model, xTrain, yTrain, xTest, yTest, 10)