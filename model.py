import numpy as np
import tensorflow as tf
import pandas as pd
import prep_data

def main():
    data = organize_data()
    data['mfcc'] = data['mfcc'].apply(eval)
    #print(data)
    X = np.array(data['mfcc'].tolist())
    print(X)
    y = np.array(data['labels'].tolist())
    model = create_model(X, y, 64, 3, (X.shape))
    model = run_model(model, X, y, 30)

def organize_data():
    # read in the data
    data = pd.read_csv('data.csv')

    return data

def create_model(X, y, neurons_per_layer, num_layers, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape[1:]))
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_model(model, X, y, num_epochs):
    model.fit(X, y, epochs=num_epochs)

if __name__ == "__main__":
    main()
    