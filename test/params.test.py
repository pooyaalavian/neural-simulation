
from src.param import ParameterSet
p = ParameterSet("defaults.json")


p.exc1.tau = 0.1
delta = p.getDelta()
print(delta)


z = p + delta
print(z.exc1)

print(z - delta)


print('bye')
