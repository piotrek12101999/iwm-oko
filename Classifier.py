from sklearn import ensemble
from joblib import dump, load
from sklearn.model_selection import cross_val_score


class Classifier:
    __clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=1)

    def __init__(self, x_train, y_train):
        self.__x_train = x_train
        self.__y_train = y_train

    def fit(self):
        self.__clf.fit(self.__x_train, self.__y_train)

    def save_model(self, name):
        dump(self.__clf, f"{name}.joblib")

    def load_model(self, path):
        self.__clf = load(path)

    def score(self):
        scores = cross_val_score(
            self.__clf,
            self.__x_train,
            self.__y_train,
            verbose=10,
            n_jobs=-1
        )

        print(f"{scores.mean()} accuracy with a standard deviation of {scores.std()}")

    def predict(self, x):
        return self.__clf.predict(x)

