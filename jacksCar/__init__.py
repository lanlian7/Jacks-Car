import time
from time import sleep

startTime  = time.time()
print(startTime)

for i in range(0,1000):
    a = i

endTime = time.time()
print(endTime)

print ('time diff:',endTime-startTime)
