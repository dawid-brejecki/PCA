

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix



df=pd.read_csv('Desktop/Wine.csv')

df.isnull().sum().sum()



df.head()


sns.set_theme(style="ticks")
sns.pairplot(df)



# wymiarow jest sporo, warto byloby sprawdzic mozliwosc ich skrocenia



Y = df.pop('Customer_Segment')

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size = 0.25)



# obowiazkowa standaryzacja
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# sprawdzenie ilosci wyjasnionej wariancji w zaleznosci od liczby wymiarow
pca = PCA().fit(X_train)


fig, ax = plt.subplots()
xi = np.arange(1, 14, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1))
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95%', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()



# z wykresu wynika, ze aby osiagnac wartosc 95% wariancji (która jest naszym celem), mozna zredukowac wymiary maksymalnie do 10



# implementacja PCA
pca = PCA(n_components = 10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

print(sum(explained_variance))



# sprawdzenie skutecznosci klasyfikacji przed i po redukcji wymiarow
classifier = KNeighborsClassifier()

classifier.fit(X_train_pca, Y_train)

y_pred = classifier.predict(X_test_pca)

cm = confusion_matrix(Y_test, y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, y_pred)}')



# klasyfikacja bez redukcji wymiarow

classifier = KNeighborsClassifier()

classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, y_pred)}')


# wyzszy wynik po PCA - dlatego ze zmienne byly skorelowane.
# przyklad ten nie ukazał pełni mozliwości redukcji wymiarów, ponieważ dane nie były zbyt liczne, stąd także
# czas ich przetwarzania nie był zauważalnie różny.
# ukazane zostało jednak, że z pomocą PCA, można skutecznie zmniejszać liczbę wymiarów, także pozbywając się 
# problemu korelacji między zmiennymi.


