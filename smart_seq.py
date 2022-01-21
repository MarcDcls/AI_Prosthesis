from tensorflow.keras import models
from data import load_data
import numpy as np

def generate_smart_seq():
    """
    Generate angles of the arm with the predictive NN on the smart sequence

    :return: outputs (angles of the arm)
    """
    predictive_NN = models.load_model("models/predictive_NN")
    data = load_data()

    result_bras = []

    for i in range(1000):
        computed_output = predictive_NN.predict(np.reshape(data[i,:12], (1, 12)))[0]
        result_bras.append(computed_output)  
        print(i)  
    #print(computed_output)
    result_bras = np.array(result_bras)
    print(result_bras.shape)    

    np.savetxt('results_bras.csv', result_bras, delimiter=',')


#generate_smart_seq()

def generate_smart_generative_seq():
    """
    Generate angles of the arm with the predictive NN on the smart generative sequence

    :return: outputs (angles of the arm)
    """
    predictive_NN = models.load_model("models/predictive_NN")
    data = load_data()

    result_bras = []
    computed_output = data[0,2:7]
    for i in range(1000):
        print(i)
        print(computed_output)
        inputs = np.concatenate((data[i,:2] , computed_output, data[i,7:12]), axis=0)
        computed_output = predictive_NN.predict(np.reshape(inputs, (1, 12)))[0]
        result_bras.append(computed_output)  
        print(computed_output)
    
    result_bras = np.array(result_bras)
    print(result_bras.shape)
    
    np.savetxt('results_bras_generative.csv', result_bras, delimiter=',') 


generate_smart_generative_seq()