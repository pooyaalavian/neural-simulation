
from src.param import ParameterSet
p = ParameterSet("defaults.json")

p.exc1.tau = 0.9
print(p.getDelta())

p.exc1.tau = 0.1
print(p.getDelta())


print('bye')
