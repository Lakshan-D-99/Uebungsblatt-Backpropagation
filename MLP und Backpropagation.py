import numpy as np # Wir brauchen numpy, damit wir Vektoren und Matritzen Berechnungen durchführen kann
import pandas as pd # Pandas, wir brauchen später, wenn wir das Iris Datensatz einlesen
import matplotlib.pyplot as plt # Damit den Fehler Diagramm darstelln

# Als die Aktivierungs funktionen sollten wir relu und sigmoid anwenden. In der Ausgangsschicht Sigmoid und in der versteckten
# Schicht relu
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Da wir Cross-Entropy als loss funktion verwenden, brauchen wir keine Ableitung von der Loss funktion zu definieren
# weil Cross-Entropy + Sigmoid = y^ - y

# Wir brauchen es für die Backpropagation
def relu_deriv(z):
    return (z > 0).astype(float)

# Klassen Namen in One-Hot-Vektoren umwandeln
def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

# MLP Klasse definieren
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        # Die Lernrate wird über das Konstruktor übergeben
        self.lr = lr

        # Gewichte initialisiren
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    # -------- Forward Propagation --------
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_hat = sigmoid(self.z2)

        return self.y_hat

    # -------- Loss berchnen mit der cross_entropy --------
    def cross_entropy(self, y, y_hat):
        eps = 1e-9
        return -np.mean(np.sum(y * np.log(y_hat + eps), axis=1))

    # -------- Backpropagation --------
    def backward(self, X, y):
        n = X.shape[0]

        # Ausgabeschicht gradient
        # dL/dz2 = y_hat - y   (CE + Sigmoid)
        dz2 = self.y_hat - y

        dW2 = (self.a1.T @ dz2) / n
        db2 = np.mean(dz2, axis=0, keepdims=True)

        # Versteckte Schicht
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_deriv(self.z1)

        dW1 = (X.T @ dz1) / n
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1



df = pd.read_csv("iris.csv", header=None)
X = df.iloc[:, 0:4].values

labels = df.iloc[:, 4].values
label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
y = np.array([label_map[l] for l in labels])

y_oh = one_hot(y, 3)

# Normalisieren
X = (X - X.mean(axis=0)) / X.std(axis=0)


mlp = MLP(input_dim=4, hidden_dim=10, output_dim=3, lr=0.05)

epochs = 2000
losses = []

for epoch in range(epochs):

    # Stochastic Gradient Descent: mischen + einzeln trainieren
    idx = np.random.permutation(len(X))

    for i in idx:
        xi = X[i:i+1]
        yi = y_oh[i:i+1]

        mlp.forward(xi)
        mlp.backward(xi, yi)

    # Loss nach kompletter Epoche berechnen
    y_hat = mlp.forward(X)
    loss = mlp.cross_entropy(y_oh, y_hat)
    losses.append(loss)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


plt.plot(losses)
plt.xlabel("Epoche")
plt.ylabel("Trainingsfehler (Cross-Entropy)")
plt.show()
