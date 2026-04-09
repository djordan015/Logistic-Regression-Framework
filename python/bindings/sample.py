import numpy as np
import time

from python.bindings import LogisticRegression

np.random.seed(42)

N_train    = 1000
N_test     = 200
n_features = 5

# Generate linearly separable data: label = 1 if sum(x) > 0 else 0
X_train = np.random.normal(size=[N_train, n_features])
y_train = (X_train.sum(axis=1) > 0).astype(float)

X_test = np.random.normal(size=[N_test, n_features])
y_test = (X_test.sum(axis=1) > 0).astype(float)

start = time.time()

model = LogisticRegression(threshold=0.5, epochs=1000, optimizer='gd', debug=True)
model.fit(X_train, y_train)

end = time.time()
print('Training time: ' + str(end - start) + ' seconds')

acc = model.score(X_test, y_test)
print('Test accuracy: ' + str(acc))
