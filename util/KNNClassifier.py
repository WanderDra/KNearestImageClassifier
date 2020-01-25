from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    def get_classifier(self):
        return self.knn

    def get_scaler(self, data_array):
        sc = StandardScaler();
        sc.fit(data_array)
        return sc

    def split(self, data_array, labels_array, test_size=0.25, random_state=42):
        data_train, data_test, labels_train, labels_test = train_test_split(data_array, labels_array, test_size=test_size, random_state=random_state, stratify=labels_array)
        return data_train, data_test, labels_train, labels_test

    def predict(self, data):
        return self.knn.predict(data)




