import joblib
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import sys



from Import_Data import *


if __name__ == "__main__":
     # "application_train.csv","application_test.csv"
     str(sys.argv)
     X,y = importation(str(sys.argv[1]),"application_test.csv")
     X = preprocess(X)


     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 7)

     with mlflow.start_run() as run :

          #mlflow.sklearn.autolog()
          clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
          clf = clf.fit(x_train, y_train)


