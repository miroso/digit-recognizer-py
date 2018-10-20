#!/Users/miroslav/anaconda3/bin/python
import csv
import pandas
import matplotlib.pyplot as plt

with open('/Users/miroslav/repos/datasets/digit_recognizer/train.csv', 'rb') as f:
    reader = csv.reader(f)

    for row in reader:
        print(row)

fpath = '/Users/miroslav/repos/datasets/digit_recognizer/train.csv'
x = pandas.io.parsers.read_csv(fpath, sep=',')
x.values[0, 1:x.shape[1]] # put variable instead of 0 to select a specific number
num = x.values[0,1:x.shape[1]].reshape(28, 28)
plt.imshow(num)
# https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting






if __name__ == "__main__":
    main()


#import tensorflow as tf
#mnist = tf.keras.datasets.mnist

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

#model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(512, activation=tf.nn.relu),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#])
#model.compile(optimizer='adam',
              #loss='sparse_categorical_crossentropy',
              #metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test, y_test)



# to do
# do file path check
