import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

#this is for single neurons 
# input_data=[1,2,3]
# weights=[[0.2,0.3,0.4],
#          [0.1,0.4,0.6],
#         [0.2,0.3,0.4]]
# bias=[2,3,4]
# output_layers=[]
# for c,ml in zip(bias , weights):
    #neuron output reset garnu parxa to find arko neuron ko output
#     neuron_output=0
#     for x,m in zip(input_data,ml):
#         neuron_output+=x*m
#     neuron_output+= c
#     output_layers.append(neuron_output)

# print(output_layers)
# this is done with numpy
# input_data=np.array([[1,2,3],[4,5,6],[6,4,3]])
# weights=np.array([[0.2,0.3,0.4],
#          [0.1,0.4,0.6],
#         [0.2,0.3,0.4]]).T

# bias=np.array([2,3,4])
# output_layer=np.dot(weights,input_data)+bias
# print(output_layer)

# yeta bata chai I am coding a class for neural network
nnfs.init()

class Rectified_Linear_Activation():
    def __init__(self):
        pass
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)


class Dense_layer():
    def __init__(self,n_input,n_neuron):
        self.weight=0.01*np.random.randn(n_input,n_neuron)
        self.bias=np.zeros((1,n_neuron))
    def forward_pass(self,inputs):
        self.output=np.dot(inputs,self.weight) +self.bias

class Softmax_Activation():
    def __init__(self):
        pass
    def forward_pass(self,inputs):
        exp=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities=exp/np.sum(exp,axis=1,keepdims=True)
        self.output=probabilities

X,y= spiral_data(samples=150,classes=3)
# Layers haru create gareko , first ma chai 2 ta input ani 4 ota neuron xa and second ma chai 4 ota input ra 4 ota nueron xa hai
dense1=Dense_layer(2,4)
dense2=Dense_layer(4,4)

# activation function haru banako
activation1=Rectified_Linear_Activation()
activation2=Softmax_Activation()

# la first neuron ma sample data gayo
dense1.forward_pass(X)
# la ayeko output lai RELU ma halyo so that it can fit non linear data
activation1.forward(dense1.output)

# second layer ma data pass garne aba
dense2.forward_pass(activation1.output)

# la aba second layer bata ayeko data lai maile second actiavation ma pathau xu
activation2.forward_pass(dense2.output)

# aba ouput layer print garne
print(activation2.output[:5])

# print(dense1.output)
# print(np.shape(X))
# print(acti_func.output[:5])
# activation2.forward_pass([[1,2,3]])
# print(activation2.output)
