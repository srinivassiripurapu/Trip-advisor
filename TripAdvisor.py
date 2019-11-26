import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame



# My Neural Network model
"""
    ---> [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX
"""



# Dimensions of each layer
"""
    --> 5 hidden layers with 30, 50, 20, 30, 9 neuron units, 
    --> A input layer with 19 input neuron units
    --> Output layer with 5 neuron units(softmax layer)
"""
layers_dims = [19, 30, 50, 20, 30, 9, 5]



# Some functions used for changing the data type 
"""
    --> Function for changing data type for YES or NO type columns
"""
def YesNoType(x):
    if x=="YES":
        return 1
    else:
        return 0



"""
    --> Function for changing data type for string type columns
"""    
def str_to_int(str):
    x=0
    for l in str:
        x += ord(l)
    return int(x)



# Reading Data
"""
    --> Reading data from the given excel sheet
    --> Division of given data into training set(80%) and test set(20%)
"""
path = ('TripAdvisor.xlsx')
df = pd.read_excel(path)

# columns in the data frame that are read from the excel sheet
cols = ['User country', 'Nr. reviews','Nr. hotel reviews','Helpful votes','Score','Period of stay','Traveler type','Swimming Pool','Exercise Room','Basketball Court','Yoga Classes','Club','Free Wifi','Hotel name','Hotel stars','Nr. rooms','User continent','Member years','Review month','Review weekday']

df['Club']=df['Club'].apply(lambda x : YesNoType(x))
df['Exercise Room']=df['Exercise Room'].apply(lambda x : YesNoType(x))
df['Swimming Pool']=df['Swimming Pool'].apply(lambda x : YesNoType(x))
df['Basketball Court']=df['Basketball Court'].apply(lambda x : YesNoType(x))
df['Free Wifi']=df['Free Wifi'].apply(lambda x : YesNoType(x))
df['Yoga Classes']=df['Yoga Classes'].apply(lambda x : YesNoType(x))

cols_2 = ['Period of stay', 'Hotel name', 'User country', 'Traveler type', 'User continent', 'Review month', 'Review weekday']

for y in cols_2:
    df[y]=df[y].apply(lambda x: str_to_int(x)) 
  
temp = df.as_matrix() # modified data frame with all columns as integer type

x = temp[:, [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
x = x.T
test_x = x[:, 0:50]
train_x = x[:, 50:505]

y = temp[:, [4]]
y = y.T
y = y-1;
y = np.array(y, dtype = 'int64')
rand = np.zeros((504, 5))
rand[np.arange(504), y] = 1
y = rand.T    
y = np.array(y, dtype = 'int64')
test_y = y[:, 0:50]
train_y = y[:, 50:505]



# Common functions used in artificial neural networks
"""
    --> Basic sigmoid function
"""
def sigmoid(z):

    s = 1/(1 + np.exp(-z))    
    return s, z



"""
    --> Basic relu function
"""
def relu(z):

    s = z*(z > 0)
    return s, z



"""
    --> Function for backward propagation
"""
def sigmoid_backward(dA, cache):

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



"""
    --> Function for backward propagation
"""
def relu_backward(dA, cache):
    
    Z = cache

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ



"""
    --> Initialising parameters
"""
def initialize_parameters_deep(layer_dims):
    """
    Input Arguments:
    layer_dims - A python list containing the dimensions of each layer in my neural network
    
    Function Returns:
    parameters - python dictionary containing my network parameters "W1", "b1", ..., "WL", "bL"
    """
    L = len(layer_dims) 
    parameters = {} 
    np.random.seed(3)      

    for l in range(1, L):
       
        # Random Initialisation
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
   
    return parameters



"""
    --> Function for forward propagation
"""
def linear_forward(A, W, b):
    """
    Input Arguments:
    A - Activations from previous layer
    W & b are weight matrix and bias vector respectively

    Function Returns:
    Z - Pre-activation parameter 
    cache -- A python dictionary containing "A", "W" and "b"
    It is stored for computing the backward propagation
    """

    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)
    return Z, cache



"""
    --> Function for forward propagation
"""
def linear_activation_forward(A_prev, W, b, activation):
    """
    Input Arguments:
    A_prev - activations from previous layer
    W & b are weight matrix and bias vector respectively
    activation - "sigmoid" or "relu" (to tell whether it is sigmoid or relu)

    Function Returns:
    A - Post-activation value 
    cache - A python dictionary containing "linear_cache" and "activation_cache";
    It is stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



"""
    --> Main function for forward propagation 
"""
def L_model_forward(X, parameters):
    """
    Input Arguments:
    parameters - Current parameters which is a dictionary cantaining weight(W) matrices and bias(b) vectors of all layers
    
    Function Returns:
    AL - last layers' post-activation value
    """

    caches = []
    A = X
    L = len(parameters) // 2         
    
    # [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    
    # [LINEAR -> SIGMOID] ---->last layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (5, X.shape[1]))
            
    return AL, caches



"""
    --> Cross-Entropy cost function
"""
def compute_cost(AL, Y, lambd, L, parameters):
    """
    Input Arguments:
    AL - last layer's activations
    Y - original output predictions

    Function Returns:
    cost - cross-entropy cost
    """
    
    m = Y.shape[1]

    temp_cost = np.dot(Y, (np.log(AL)).T) + np.dot(1-Y, (np.log(1-AL)).T)
    temp_cost = temp_cost * np.eye(np.shape(AL)[0], dtype = 'int64')
    cost = (-1/m) * np.sum(temp_cost)

    reg_cost = 0

    for l in range(1, L):
        reg_cost = reg_cost + np.sum(np.square(parameters['W' + str(l)]))

    cost = cost + (lambd/(2*m)) * reg_cost   
    cost = np.squeeze(cost)    
    assert(cost.shape == ())
    
    return cost



"""
    --> Function for backward propagation
"""
def linear_backward(dZ, cache, lambd):
    """
    Input Arguments:
    dZ - Gradient of the cost with respect to the linear output of current layer l
    cache - tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Function Returns:
    dA_prev - Gradient of the cost with respect to the activation of the previous layer l-1
    dW - Gradient of the cost with respect to W for current layer l
    db - Gradient of the cost with respect to b for current layer l
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T) + (lambd/m) * W
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db



"""
    --> Function for backward propagation
"""
def linear_activation_backward(dA, cache, activation, lambd):
    """
    Input Arguments:
    dA - post-activation gradient for current layer l 
    cache - tuple of values (linear_cache, activation_cache)
    
    Function Returns:
    dA_prev - Gradient of the cost with respect to the activation of the previous layer l-1
    dW -- Gradient of the cost with respect to W for current layer l
    db -- Gradient of the cost with respect to b for current layer l
    """

    linear_cache, activation_cache = cache
    
    if activation == "relu":
   
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
     
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db



"""
    --> Main function for backward propagation
"""
def L_model_backward(AL, Y, caches, lambd):
    """
    Function Returns:
    grads - A dictionary with the gradients containing dA, dW, db
    """
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    
    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid', lambd)
    
    # Loop from l = L-2 to l = 0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients. 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, 'relu', lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



"""
    --> for updating parameters at each gradient step
"""
def update_parameters(parameters, grads, learning_rate):
    """
    Input Arguments:
    parameters - python dictionary containing all parameters 
    grads - python dictionary containing your gradients, output of L_model_backward
    """
    
    L = len(parameters) // 2 

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters



"""
    --> Final model
"""
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0065, num_iterations = 3000, print_cost=False, lambd = 0.7):
    
    # keeping track of cost
    costs = []                     
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)
    
    # Gradient descent
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y, lambd, len(layers_dims), parameters)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches, lambd)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Printing the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



"""
    --> calling for the main function to get the final parameters
"""
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 7200, print_cost = True, lambd = 0.7)



"""
    --> for predicting the output and calculating accuracy
"""
def predict(X, Y, parameters):
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
 
    AL, cache = L_model_forward(X, parameters)
    predictions = np.zeros(np.shape(AL))

    for i in range(np.shape(Y)[1]):
        col_max = np.max(AL[:, i])
        predictions[:, i] = (AL[:, i] == col_max)    

    temp_accuracy = np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T)
    temp_accuracy = temp_accuracy * np.eye(np.shape(AL)[0], dtype = 'int64')
    accuracy = (np.sum(temp_accuracy)/(Y.size))*100

    return accuracy, predictions



"""
    --> printing predicted values for training set
"""
print("-----------------------------------------------------------------------------\n")
train_accuracy, pred_train = predict(train_x, train_y, parameters)
print("The predicted values for training set---\n")
print(pred_train)
print("-----------------------------------------------------------------------------\n")



"""
    --> printing predicted values for test set
"""
print("-----------------------------------------------------------------------------\n")
test_accuracy, pred_test = predict(test_x, test_y, parameters)
print("The predicted values for test set---\n")
print(pred_test)
print("-----------------------------------------------------------------------------\n")



"""
    --> printing train_accuracy and test_accuracy
"""
print("The Accuracy for the training set is : ")
print(train_accuracy)
print("-----------------------------------------------------------------------------\n")
print("The Accuracy for the test set is : ")
print(test_accuracy)
print("-----------------------------------------------------------------------------\n")