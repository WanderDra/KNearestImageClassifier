from sklearn.neighbors import KNeighborsClassifier
from util import Readrer as reader


class Classifier:
    knn = KNeighborsClassifier()

    def classify(self):
        table = reader.read_file('data_mnist.csv')
        length = len(table[1])
        print(table[1][1:length])
        pass


def main():
    c = Classifier()
    c.classify()


if __name__ == '__main__':
    main()