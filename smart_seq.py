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
        computed_output = predictive_NN.predict(np.reshape(data[i], (1, 17)))[0]
        result_bras.append(computed_output)  
        print(i)  
    #print(computed_output)
    result_bras = np.array(result_bras)
    print(result_bras.shape)    

    np.savetxt('results_bras.csv', result_bras, delimiter=',')


generate_smart_seq()    