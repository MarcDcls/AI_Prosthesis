from tensorflow.keras import models
from data import load_data
import numpy as np

def generate_smart_seq():
    """
    Generate angles of the arm with the predictive NN during the sequence based on the last real configuration of the
    arm for each prediction

    :return: outputs (angles of the arm)
    """
    predictive_NN = models.load_model("models/predictive_NN")
    data = load_data()

    result_bras = []

    for i in range(1000):
        computed_output = predictive_NN.predict(np.reshape(data[i], (1, 17)))[0]
        result_bras.append(computed_output)  
        print(i)  
    #print(computed_output)
    result_bras = np.array(result_bras)
    print(result_bras.shape)    

    np.savetxt('results_bras.csv', result_bras, delimiter=',')


def generate_smart_generative_seq():
    """
    Generate angles of the arm with the predictive NN during the sequence by generating the prediction based only on
    the first configuration of the arm

    :return: outputs (angles of the arm)
    """
    predictive_NN = models.load_model("models/predictive_NN")
    data = load_data()

    result_bras = []
    computed_output = data[0,6:11]
    for i in range(1000):
        inputs = np.concatenate((data[i,:7] , computed_output, data[i,12:]), axis=0)
        computed_output = predictive_NN.predict(np.reshape(inputs, (1, 17)))[0]
        result_bras.append(computed_output)  
    
    result_bras = np.array(result_bras)
    print(result_bras.shape)
    
    np.savetxt('results_bras_generative.csv', result_bras, delimiter=',') 

