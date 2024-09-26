import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

class Dense_layer():
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
    def forward_pass(self,inputs):
        self.inputs=inputs
        self.output=np.dot(self.inputs,self.weights)+self.bias
    def backward_pass(self,dvalues):
        self.dinputs=np.dot(dvalues,self.weights.T)
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbias=np.sum(dvalues,axis=0,keepdims=True)

class Relu_func():
    def forward_pass(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,inputs)
    def backward_pass(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<=0]=0

class Softmax_func():
    def forward_pass(self,inputs):
        exp=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities=exp/np.sum(exp,axis=1,keepdims=True)
        self.output=probabilities
    def backward_pass(self,dvalues):
        self.dinputs=np.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output=single_output.reshape(-1,1)
            jacobian_matrix=np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index]=np.dot(jacobian_matrix,single_dvalues)
class Adam_optimizer():
    def __init__(self,learning_rate=0.001,decay_rate=0,epsilon=1e-7,beta_1=0.9,beta_2=0.99):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay_rate
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.iteration=0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*1/(1+self.iteration*self.decay)
    def update_params(self,layer):
        if not hasattr(layer,"weight_cache"):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.weight_momentum=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.bias)
            layer.bias_momentum=np.zeros_like(layer.bias)
        layer.weight_momentum=self.beta_1*layer.weight_momentum-layer.dweights*(1-self.beta_1)
        layer.bias_momentum=self.beta_1*layer.bias_momentum-layer.dbias*(1-self.beta_1)

        weight_momentum_corrected=layer.weight_momentum/(1-self.beta_1**(self.iteration+1))
        bias_momentum_corrected=layer.bias_momentum/(1-self.beta_1**(self.iteration+1))

        layer.bias_cache=self.beta_2*layer.bias_cache +(1-self.beta_2)*layer.dbias**2
        layer.weight_cache=self.beta_2*layer.weight_cache + (1-self.beta_2)*layer.dweights**2

        bias_cache_corrected=layer.bias_cache/(1-self.beta_2**(self.iteration+1))
        weight_cache_corrected=layer.weight_cache/(1-self.beta_2**(self.iteration+1))

        layer.weights+=self.current_learning_rate*weight_momentum_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.bias+=self.current_learning_rate*bias_momentum_corrected/(np.sqrt(bias_cache_corrected)+self.epsilon)

    def post_update_params(self):
        self.iteration+=1
    
class Loss():
    def calculate(self,y_pred,y_true):
        loss=self.forward_pass(y_pred,y_true)
        loss_mean=np.mean(loss)
        return loss_mean
class Categorical_loss_func(Loss):
    def forward_pass(self,output,y):
        samples=len(output)
        y_pred_clipped=np.clip(output,1e-7,1-1e-7)
        if len(y.shape)==1:
            confidence=y_pred_clipped[range(samples),y]
        elif len(y.shape)==2:
            confidence=np.sum(y_pred_clipped*y,axis=1,keepdims=True)
        negative_loss=-np.log(confidence)
        return negative_loss
    def backward_pass(self,dvalues,y):
        samples=len(dvalues)
        labels=len(dvalues[0])
        if len(y.shape)==1:
            y=np.eye(labels)[y]
        self.dinputs=-y/dvalues
        self.dinputs=self.dinputs/samples

class Activation_Categorical():
    
    def __init__(self):
        self.activation=Softmax_func()
        self.loss_func=Categorical_loss_func()
    
    def forward_pass(self,inputs,y):
        self.activation.forward_pass(inputs)
        self.output=self.activation.output
        return self.loss_func.calculate(self.output,y)
    
    def backward_pass(self,dvalues,y):
        samples=len(dvalues)
        self.dinputs=dvalues.copy()
        if len(y.shape)==2:
            y=np.argmax(y,axis=1)
        self.dinputs[range(samples),y]-=1
        self.dinputs=self.dinputs/samples

layer1=Dense_layer(2,64)
layer2=Dense_layer(64,3)
activation1=Relu_func()
loss_activation=Activation_Categorical()
optimize=Adam_optimizer(learning_rate=0.05, decay_rate=5e-7)

X,y=spiral_data(classes=3,samples=750)


for epoch in range(10001):
# forward_pass
    layer1.forward_pass(X)
    
    activation1.forward_pass(layer1.output)
    layer2.forward_pass(activation1.output)
  
    loss=loss_activation.forward_pass(layer2.output,y)


    confidence_score=np.argmax(loss_activation.output,axis=1)
    if len(y.shape)==2:
        y=np.argmax(y,axis=1)
    acc=np.mean(confidence_score==y)
    # print(loss_activation.output)
    if not epoch %100:
        print(f'epoch: {epoch:.3f},'+
            f'acc :{acc:.3f},'+
            f'loss :{loss:.3f},'+
            f'lr :{optimize.current_learning_rate}')
    
    # backward_pass
    loss_activation.backward_pass(loss_activation.output,y)
    layer2.backward_pass(loss_activation.dinputs)
    activation1.backward_pass(layer2.dinputs)
    layer1.backward_pass(activation1.dinputs)

    optimize.pre_update_params()
    optimize.update_params(layer1)
    optimize.update_params(layer2)
    optimize.post_update_params()