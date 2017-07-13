import pandas as pd
import matplotlib.pyplot as plt
a = {'Test1': {1: 21867186, 4: 20145576, 10: 18018537},
    'Test2': {1: 23256313, 4: 21668216, 10: 19795367}}

d = pd.DataFrame(a).T
#print d

f = plt.figure()

plt.title('Title here!', color='black')
d.plot(kind='bar', ax=f.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()