import pandas as pd
import lib

FILENAME = "iris.csv"
HEADERS = ['a','b','c','d', 'class']

df = pd.read_csv(FILENAME, names=HEADERS)
print(df.sample(8))

# Factorize the data
df['class'], index = pd.factorize(df['class'])

# Split the data into train and test partitions
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = lib.train_test_split(X, y)

model = lib.Perceptron()

model.train(X_train, y_train)

model.test(X_test, y_test)

print("done")
