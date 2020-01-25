from sklearn.neighbors import KNeighborsClassifier
from util import Readrer as reader
import numpy as np
from util import KNNClassifier
from util import ModelSaveTools as saver
from util import ImageTool as it

knn = KNNClassifier.KNNClassifier()


def training_data_collect():
    table, row_num = reader.read_file('data_mnist.csv')
    length = len(table[1])
    data = []
    labels = []
    for i in range(1, row_num):
        data.append(table[i][1:length])
        labels.append(table[i][0])
    data_array = np.array(data)
    labels_array = np.array(labels)
    print(data_array)
    print(labels_array)
    return data_array, labels_array


def testing_data_collect():
    table, row_num = reader.read_file('test_mnist.csv')
    length = len(table[1])
    data = []
    for i in range(1, row_num):
        data.append(table[i][0:length])
    data_array = np.array(data)
    print(data_array)
    return data_array


def train():
    data_array, labels_array = training_data_collect()
    knn_c = knn.get_classifier()
    data_train, data_test, labels_train, labels_test = knn.split(data_array, labels_array)
    sc = knn.get_scaler(data_train)
    sc_data_train = sc.transform(data_train)
    sc = knn.get_scaler(data_test)
    sc_data_test = sc.transform(data_test)
    knn_c.fit(sc_data_train, labels_train)
    print(knn_c.score(sc_data_train, labels_train))
    print(knn_c.score(sc_data_test, labels_test))
    saver.save(knn_c, 'TrainedKNNModel')


def predict():
    knn_c = saver.load('TrainedKNNModel')
    data_array, labels_array = training_data_collect()

    test_data = testing_data_collect()

    data = test_data[5486]
    it.draw(data)

    sc = knn.get_scaler(test_data)
    sc_data = sc.transform(test_data)
    # print(knn_c.score(sc_data, labels_array))

    print(knn_c.predict([sc_data[5486]]))



# train()
predict()


# class Classifier:
#     knn = KNeighborsClassifier(n_jobs=-1)
#
#     def classify(self):
#         table = reader.read_file('data_mnist.csv')
#         length = len(table[1])
#         print(table[1][1:length])
#         pass
#
#
#
# def main():
#     c = Classifier()
#     c.classify()


# if __name__ == '__main__':
#     main()