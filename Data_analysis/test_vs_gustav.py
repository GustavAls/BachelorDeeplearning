
import numpy as np
import datetime
num_times = int(1e6)
start = datetime.datetime.now()
for i in range(num_times):
    x = np.random.normal((100,100)) * np.random.normal((100,100))
slut = datetime.datetime.now()

print(slut-start)