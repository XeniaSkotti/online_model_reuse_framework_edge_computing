import matplotlib.pyplot as plt
from node import get_node_data

def visualise_experiment(node_data):
    plt.rcParams["figure.figsize"] = (8,5)
    
    a,b,c,d = node_data
    
    plt.scatter(a.humidity, a.temperature, marker= ".", label="pi2", alpha =0.5)
    plt.scatter(b.humidity, b.temperature, marker = "v", label="pi3", alpha=0.5)
    plt.scatter(c.humidity, c.temperature, marker = "^", label="pi4", alpha=0.5)
    plt.scatter(d.humidity, d.temperature, marker = "*", label="pi5", alpha=0.5)
    plt.legend()
    plt.xlabel(xlabel="Humidity")
    plt.ylabel(ylabel="Temperature")
    plt.show()

def visualise_experiments(data):
    for i in range(1,4):
        node_data = get_node_data(data, experiment = i)
        visualise_experiment(node_data)