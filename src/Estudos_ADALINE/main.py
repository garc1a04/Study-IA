import numpy as np
from neural_network import ADALINE

X = np.array([
[1,1],
[0,1],
[0,2],
[1,0],
[2,2],
[4,1.5],
[1.5,6],
[3,5],
[3,3],
[6,4  ],
])

y = np.array([
[1],
[1],
[1],
[1],
[1],
[-1],
[-1],
[-1],
[-1],
[-1]
])

ps = ADALINE(X.T, y.T, 1e-3)
ps.fit(epsilon=0.0001)