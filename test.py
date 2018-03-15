from matplotlib import pyplot as plt
import generator as gen

g = gen.generate(10)

print(next(g))

x1_2, x3 = next(g)
x3 = x3.reshape(x3.shape[1], x3.shape[2])
print(x3.shape)
plt.imshow(x3)
plt.show()
