import matplotlib.pyplot as plt
from node import get_node_data

def visualise_train_test_data(node_data):
    plt.rcParams["figure.figsize"] = (9,5)
    
    a,b,c,d = node_data
    
    plt.scatter(a[0].humidity, a[0].temperature, marker= ".", label="pi2-train", alpha =0.3)
    plt.scatter(b[0].humidity, b[0].temperature, marker = "v", label="pi3-train", alpha=0.3)
    plt.scatter(c[0].humidity, c[0].temperature, marker = "<", label="pi4-train", alpha=0.3)
    plt.scatter(d[0].humidity, d[0].temperature, marker = "*", label="pi5-train", alpha=0.3)
    
    plt.scatter(a[1].humidity, a[1].temperature, marker= "o", label="pi2-test", alpha =0.3)
    plt.scatter(b[1].humidity, b[1].temperature, marker = "^", label="pi3-test", alpha=0.3)
    plt.scatter(c[1].humidity, c[1].temperature, marker = ">", label="pi4-test", alpha=0.3)
    plt.scatter(d[1].humidity, d[1].temperature, marker = "+", label="pi5-test", alpha=0.3)
    
    plt.legend()
    plt.xlabel(xlabel="Humidity")
    plt.ylabel(ylabel="Temperature")
    plt.show()
    
    

def visualise_experiment(node_data):
    plt.rcParams["figure.figsize"] = (8,5)
    
    a,b,c,d = node_data
    
    plt.scatter(a.humidity, a.temperature, marker= ".", label="pi2", alpha =0.5)
    plt.scatter(b.humidity, b.temperature, marker = "v", label="pi3", alpha=0.5)
    plt.scatter(c.humidity, c.temperature, marker = "<", label="pi4", alpha=0.5)
    plt.scatter(d.humidity, d.temperature, marker = "*", label="pi5", alpha=0.5)
    
    plt.legend()
    plt.xlabel(xlabel="Humidity")
    plt.ylabel(ylabel="Temperature")
    plt.show()

def visualise_experiments(data):
    for i in range(1,4):
        node_data = get_node_data(data, experiment = i)
        visualise_experiment(node_data)