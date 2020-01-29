from sklearn.neighbors import KNeighborsClassifier
from util import Readrer as reader
import numpy as np
from util import KNNClassifier
from util import ModelSaveTools as saver
from util import ImageTool as it
import matplotlib.pyplot as plt

knn = KNNClassifier.KNNClassifier()


# read training data
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


# read testing data
def testing_data_collect():
    table, row_num = reader.read_file('test_mnist.csv')
    length = len(table[1])
    data = []
    for i in range(1, row_num):
        data.append(table[i][0:length])
    data_array = np.array(data)
    print(data_array)
    return data_array


# train KNN network
def train(k):
    data_array, labels_array = training_data_collect()
    knn_c = knn.get_classifier()
    knn_c.n_neighbors = k
    data_train, data_test, labels_train, labels_test = knn.split(data_array, labels_array)
    # sc = knn.get_scaler(data_train)
    # sc_data_train = sc.transform(data_train)
    # sc = knn.get_scaler(data_test)
    # sc_data_test = sc.transform(data_test)
    
    sc_data_train = data_train
    sc_data_test = data_test


    knn_c.fit(sc_data_train, labels_train)

    knn_graph = knn_c.kneighbors_graph(data_test, k, mode='distance')
    print(knn_graph)
    f = open('NeighborDistance_test.txt', 'w')
    f.write(str(knn_graph))
    f.close()

    print(knn_c.score(sc_data_train, labels_train))
    print(knn_c.score(sc_data_test, labels_test))
    saver.save(knn_c, 'TrainedKNNModel')


# manually check
def predict(id):
    knn_c = saver.load('TrainedKNNModel')
    # data_array, labels_array = training_data_collect()

    test_data = testing_data_collect()

    data = test_data[id]
    it.draw(data)

    sc = knn.get_scaler(test_data)
    sc_data = sc.transform(test_data)
    # print(knn_c.score(sc_data, labels_array))

    print(knn_c.predict([sc_data[id]]))


# accuracy test and get actual incorrect result
def test():
    knn_c = saver.load('TrainedKNNModel')
    data_array, labels_array = training_data_collect()

    test_data = data_array
    test_labels = labels_array

    sc = knn.get_scaler(test_data)
    sc_data = sc.transform(test_data)

    print(test_labels)

    count = 0
    correct = 0
    result = knn_c.predict(sc_data)
    print(result)
    incorrect = []
    for data in result:
        if data == test_labels[count]:
            correct += 1
        else:
            incorrect.append(count)
        count += 1
    accuracy = correct / len(sc_data)

    print(accuracy)

    # record incorrect data
    f = open('Record.txt', 'w')
    f.write(str(incorrect))
    f.close()


# final output
def final_result():
    knn_c = saver.load('TrainedKNNModel')
    test_data = testing_data_collect()

    sc = knn.get_scaler(test_data)
    sc_data = sc.transform(test_data)

    result = knn_c.predict(sc_data)
    id = 1
    count = 0
    filenum = 1
    f = open('result' + str(filenum) + '.txt', 'w')
    f.writelines('ImageId,Label\n')
    for data in result:
        if count < 10000:
            f.writelines(str(id) + ',' + str(data) + '\n')
            id += 1
            count += 1
        else:
            filenum += 1
            f.close()
            f = open('result' + str(filenum) + '.txt', 'w')
            f.writelines('ImageId,Label\n')
            count = 0
    f.close()

    knn_graph = knn_c.kneighbors_graph(sc_data, 5, mode='distance')
    print(knn_graph)

    f = open('NeighborDistance_test.txt', 'w')
    f.write(str(knn_graph))
    f.close()



## actions
train(5)
# predict(7331)
# test()
# final_result()