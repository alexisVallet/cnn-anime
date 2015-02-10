import matplotlib.pyplot as plt
from numpy.random import rand
from numpy import arange

val = 3-6*rand(5)    # the bar lengths
pos = arange(5)+.5    # the bar centers on the y axis
print pos

fig = plt.figure()
ax = fig.add_subplot(111)
ax.barh(pos,val, align='center',height=0.1)
ax.set_yticks(pos, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))

ax.axvline(0,color='k',lw=3)   # poor man's zero level

ax.set_xlabel('Performance')
ax.set_title('horizontal bar chart using matplotlib')
ax.grid(True)
plt.show()
