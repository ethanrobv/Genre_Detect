import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp

#-------------Data Prep-----------------#

#load data from data.csv
data=np.loadtxt(fname='Genre_Detect\data.csv',delimiter=',',dtype=float,skiprows=1)

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

#----------------BNN----------------#

def run_model(model, x_train, y_train, x_test, y_test, num_epochs):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    model.fit(x_train, y_train, batch_size=2, epochs=num_epochs)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

def create_bnn_model(train_size):
    inputs = keras.Input(shape=(int(len(xTrain[0])),))

    dense = layers.Dense(300, activation="relu")(inputs)

    dense = layers.Dense(300, activation="relu")(dense)

    outputs = layers.Dense(2)(dense)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


bnn_model = create_bnn_model(len(xTrain))
bnn_model.summary()
run_model(bnn_model, xTrain, yTrain, xTest, yTest, 100)