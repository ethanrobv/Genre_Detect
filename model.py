import numpy as np
import tensorflow as tf
import pandas as pd
import prep_data
# import train_test_split from sklearn
from sklearn.model_selection import train_test_split

def main():
    data = organize_data()
    data['mfcc'] = data['mfcc'].apply(eval)
    #print(data)
    X = np.array(data['mfcc'].tolist())
    #print(X)
    y = np.array(data['labels'].tolist())
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = create_model(64, 3, (X.shape))
    model = run_model(model, X_train, y_train, 100)

def organize_data():
    # run the prep_data.py file
    prep_data.main()
    # read in the data
    data = pd.read_csv('data.csv')

    return data

def create_model(neurons_per_layer, num_layers, input_shape):
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
    