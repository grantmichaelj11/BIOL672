# -*- coding: utf-8 -*-
"""
@author: Grant

This script is for optimizing neural networks on a compute cluster to assess:
    
1) How the number of hidden layers impacts performance
2) How the number of CPU cores used increases speed

    
Note: I did not use CUDA because I do not use a NVIDA graphics card. Instead I
parellelized on my CPU. This entire script will most likely take you 10ish minutes
to run completely


Operating System: Windows 10/11
Packages: time, sklearn, torch, numpy, matplotlib
Data Files: NBA_gammas.csv
"""
import time
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, layer_size, output_size, num_layers):
        
        super(NeuralNetwork, self).__init__()
        
        self.input_layer = nn.Linear(input_size, layer_size)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_size, layer_size) for _ in range(num_layers)
            ])
        
        self.output_layer = nn.Linear(layer_size, output_size)
        
    def forward(self, x):
        
        x = torch.relu(self.input_layer(x))
        
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
            
        output = self.output_layer(x)
        
        return output

def train_neural_network(processors, X, y, model, n_epochs):
    
    torch.manual_seed(processors)
    
    torch.set_num_threads(processors)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(n_epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    accuracy = loss.item()
    
    return accuracy

def main(cpus, layers, n_epochs):
    
    torch.set_flush_denormal(True)
    
    california_housing = fetch_california_housing()
    
    X = torch.from_numpy(california_housing.data).float()
    y = torch.from_numpy(california_housing.target).float()
    
    input_size = len(X[0])
    layer_size = input_size*2
    output_size = 1
    
    model = NeuralNetwork(input_size, layer_size, output_size, layers)
    
    return train_neural_network(cpus, X, y, model, n_epochs)
    

if __name__ == '__main__':
    
    #Test of number of processors vs training time
    processors_to_test = 16
    processor_tracker = np.arange(1,processors_to_test+1)
    time_history = []

    for i in range(processors_to_test):
        start_time = time.time()
        test = main(i+1,1,10)
        end_time = time.time()
        
        elapsed_time = end_time-start_time
        
        time_history.append(elapsed_time)
        
    plt.plot(processor_tracker, time_history, marker ='o', linestyle='-')
    plt.xlabel('Number of Processors')
    plt.ylabel('Training Time (s)')
    
    plt.savefig('neural_network_plots/Training_Time_vs_Processors_Neural_Network.png')
    plt.show()
    
    
    
    opt_processors = time_history.index(min(time_history)) + 1
    
    layers_to_test = 8
    layer_tracker = np.arange(1, layers_to_test+1)
    layer_history = []
    
    for i in range(layers_to_test):
        
        test=main(opt_processors, i, 50)
        
        layer_history.append(test)
        
    plt.plot(layer_tracker, layer_history, marker ='o', linestyle='-')
    plt.xlabel('Number of Layers')
    plt.ylabel('MSE')
    
    plt.savefig('neural_network_plots/layers_vs__Neural_Network.png')
    
    plt.show()
    
"""
As the number of processors increases the training time decreases. It appears that
this minimizes at roughly 7/8 processors. Above this number it appears that my current
computer begins to battle for resources and performance begins to reduce.

Given 50 training epochs (for sakes of time) It appears that an optimal number of
layers to reduce our mean squared error is 4. Higher numbers of layers begin to see
a drastic increase, and lower layers also see a significant increase in error.
"""
    

