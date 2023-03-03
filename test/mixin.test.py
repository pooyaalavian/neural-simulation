from src.mixin import A

x1 = A({'a':1,'b':2,'c':3})
x2 = A({'a':3,'b':2,'c':1})

a = A({'x':x1, 'y':5})
b = A({'x':x2, 'y':5})


print('a')
print (a.__json__())
delta = a - b # a minus values of b that match
print('delta = a-b =')
print(delta)

aa = b+delta
print('b+delta')
print(aa)
print(a)
print(aa==a)
print('bye')