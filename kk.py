import random

from tabulate import tabulate

m = [['rar', random.randint(0, 12), random.random()],
     ['rar', random.randint(0, 12), random.random()],
     ['rar', random.randint(0, 12), random.random()],
     ['rar', random.randint(0, 12), random.random()],
     ['rar', random.randint(0, 12), random.random()], ]
m.append(['rar', random.randint(12, 32), random.random()])
print(tabulate(m, tablefmt="fancy_grid"))
p = []
for element in m:
    print(element)
    p.append(element)
print(tabulate(p, tablefmt="html"))
