import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data.csv', names=names)
X = np.array(df.iloc[:, 0:4])
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(X_train, y_train)
#OR
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

pred = knn.predict(X_test)
print('Model accuracy score: ', accuracy_score(y_test, pred))

print('Index\tPredicted\tActual')
for i in range(len(pred)):
    if pred[i]!=y_test[i]:
        print(i, '\t', pred[i], '\t', y_test[i], '***')

DataToPredict = np.array([[5.2, 3.5, 1.4, 0.2], [5.7, 2.9, 3.6, 1.3], [5.8, 3.0, 5.1, 1.8]])
pred = knn.predict(DataToPredict)

print('Predicted Results\n')
for i in range(len(pred)):
    print('\t', DataToPredict[i], '\t', pred[i])

for i in range(len(y)):
    if y[i] == 'Iris-setosa':
        y[i] = 1
    elif y[i] == 'Iris-versicolor':
        y[i] = 2
    else:
        y[i] = 3

plt.scatter(df['sepal_length'], df['sepal_width'], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

plt.scatter(df['petal_length'], df['petal_width'], c=y)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
