import numpy as np

# DATATYPES AND ATTRIBUTES

'''
Possuimos 3 tipos de numpy

Vetor -> [1 2 3 4]
Matriz: 
[
    [1 2 3 4],
    [1 2 3 4]
]

Matriz com n dimensões
[
    [
        [1 2 3 4],
        [1 2 3 4]
    ],
    [
        [1 2 3 4],
        [1 2 3 4]
    ],
]
'''

a1 = np.array([1,2,3,4])
print(a1.dtype)
print(type(a1))
print(a1.shape)
print(a1.size)

a1 = np.array([[1,2,3,4], [1,2,3,4]])
print(a1.dtype)
print(type(a1))
print(a1.shape)
print(a1.size)
print(f"dimensão: {a1.ndim}")

a1 = np.array([[[1,2,3,4], [1,2,3,4]],[[1,2,3,4], [1,2,3,4]]])
print(a1.dtype)
print(type(a1))
print(a1.shape)
print(a1.size)
print(f"dimensão: {a1.ndim}")

'''
Exercicios 

Crie:

- Um vetor de 10 zeros

- Um vetor de 10 uns

- Um vetor de 0 a 9

- Um array 3x3 com números de 0 a 8
'''

ex1 = np.zeros(10)
print(ex1)

ex1 = np.ones(10)
print(ex1)

ex1 = np.arange(0, 10, 1)
print(ex1)

print("--------------------------------------------------------")
ex1 = np.arange(0, 9).reshape((3,3))
print(ex1)

print("--------------------------------------------------------")
ex1 = np.arange(3).reshape(1, 3)
resultado = ex1 + np.zeros((3,1))

print(resultado)


print("--------------------------------------------------------")
# NÍVEL 2 — Broadcasting
# Normalização

a = np.random.randint(1, 100, (5,5))
max = a.max()
min = a.min()
num = a - min
den = max-min

resultado = num / den
print(resultado)


print("--------------------------------------------------------")
# Subtração linha a linha

a = np.random.rand(4,3)
b = np.random.rand(3)

print(b - a)
print(a)
print(b)

print("--------------------------------------------------------")
# NÍVEL 3 — Broadcasting
# Álgebra Linear

mat_a = np.arange(0, 9).reshape((3,3))
mat_b = np.arange(0, 18, 2).reshape((3,3))

print(mat_a)
print(mat_b)

# Produto

resultado = mat_a @ mat_b
print(resultado)

#Transposta

print(mat_a.T)
print(mat_b.T)

#Determinante

mat_random = np.random.randint(0, 12, size=(12,12))
resultado = np.linalg.det(mat_random)
print(resultado)