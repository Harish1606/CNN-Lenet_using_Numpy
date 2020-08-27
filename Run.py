"""#importing packages
import numpy as np

from lenet import Lenet
#input
np.random.seed(0)

x=np.random.uniform(size=(3,32,32))
y=np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]])
 
emp=Lenet.function(x,y)
print(emp)"""
import pyaudio

