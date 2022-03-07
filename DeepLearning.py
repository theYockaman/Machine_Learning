import numpy as np
from Functions import Data

class DeepLearning:
    def __init__(self,Inputs:list, Number_Outputs:int , Directory:str, Target:list = None):
        self.Inputs = Inputs
        self.Target = Target
        self.Ouputs = []

        self.Directory = Directory

        # Calculate Layers and Number of Inputs and Number of Outputs
        self.Number_Inputs = len(np.array(Inputs))
        self.Number_Outputs = Number_Outputs

        # Number of Hidden Layers with Number of Output Layers
        self.Number_Layers = 5
        self.NueronsPerLayer = round(((self.Number_Inputs * 2)/3) + self.Number_Outputs)
        
        # Create Weights, Biases, Nuerons, and Errors
        self.Weights = self.Intialize_Weights()
        self.Biases = self.Intialize_Biases()
        self.Nuerons = [np.array([self.Inputs])]
        self.Errors = []

        # Constants
        self.A = 0.001
        self.B1 = 0.9
        self.B2 = 0.999
        self.Bias_Prev_M = self.Intialize_Bias_M()
        self.Bias_Prev_V = self.Intialize_Bias_V()
        self.Weight_Prev_M = self.Intialize_Weight_M()
        self.Weight_Prev_V = self.Intialize_Weight_V()

    def Intialize_Weights(self):
        file = Data.JSON_Data(self.Directory)

        if file.get("Weights"):
            weights = Data.Convert_Array(file['Weights'])
        else:
            weights = []
                
            for layer in range(self.Number_Layers):

                if layer == 0:
                    intial = self.Number_Inputs 
                    final = self.NueronsPerLayer
                elif layer == (self.Number_Layers-1):
                    intial = self.NueronsPerLayer
                    final = self.Number_Outputs
                else:
                    intial = self.NueronsPerLayer
                    final = self.NueronsPerLayer

                layer_weights = np.random.random((intial,final))
                weights.append(layer_weights)


        return weights

    def Intialize_Biases(self):
        file = Data.JSON_Data(self.Directory)

        if file.get("Biases"):
            
            biases = Data.Convert_Array(file['Biases'])

        else:
            biases = []

            for layer in range(self.Number_Layers):

                if layer == (self.Number_Layers-1):
                    final = self.Number_Outputs
                else:
                    final = self.NueronsPerLayer

                list_biases = np.ones((1, final))[0]
                biases.append(list_biases)

        return biases
    
    def Intialize_Weight_M(self):
        file = Data.JSON_Data(self.Directory)

        if file.get("Weight M"):
            m_s = Data.Convert_Array(file['Weight M'])
        else:
            m_s = []
                
            for layer in range(self.Number_Layers):

                if layer == 0:
                    intial = self.Number_Inputs 
                    final = self.NueronsPerLayer
                elif layer == (self.Number_Layers-1):
                    intial = self.NueronsPerLayer
                    final = self.Number_Outputs
                else:
                    intial = self.NueronsPerLayer
                    final = self.NueronsPerLayer

                layer_m_s = np.zeros((intial,final))
                m_s.append(layer_m_s)

        return m_s

    def Intialize_Weight_V(self):
        file = Data.JSON_Data(self.Directory)

        if file.get("Weight V"):
            v_s = Data.Convert_Array(file['Weight V'])
        else:
            v_s = []
                
            for layer in range(self.Number_Layers):

                if layer == 0:
                    intial = self.Number_Inputs 
                    final = self.NueronsPerLayer
                elif layer == (self.Number_Layers-1):
                    intial = self.NueronsPerLayer
                    final = self.Number_Outputs
                else:
                    intial = self.NueronsPerLayer
                    final = self.NueronsPerLayer

                layer_v_s = np.zeros((intial,final))
                v_s.append(layer_v_s)

        return v_s

    def Intialize_Bias_M(self):
        file = Data.JSON_Data(self.Directory)

        if file.get("Biases M"):
            print("WHY")
            v_m = Data.Convert_Array(file["Biases M"])

        else:
            v_m = []

            for layer in range(self.Number_Layers):

                if layer == (self.Number_Layers-1):
                    final = self.Number_Outputs
                else:
                    final = self.NueronsPerLayer
            
                layer_v_m = np.zeros((1,final))[0]
                v_m.append(layer_v_m)

        return v_m

    def Intialize_Bias_V(self):

        file = Data.JSON_Data(self.Directory)

        if file.get("Biases V"):
            v_s = Data.Convert_Array(file["Biases V"])

        else:
            v_s = []

            for layer in range(self.Number_Layers):

                if layer == (self.Number_Layers-1):
                    final = self.Number_Outputs
                else:
                    final = self.NueronsPerLayer
            
                layer_v_s = np.zeros((1,final))[0]
                v_s.append(layer_v_s)

        return v_s

    def Sigmoid_Activation(self,Inputs):
        sigmoid = 1/(1 + np.exp(-Inputs))
        return sigmoid

    def ReLU_Activation(self,Inputs):
        ReLU = np.maximum(0,Inputs)
        return ReLU

    def Calculate_Layer(self,Inputs,Weights,Biases):
        layer = np.dot(Inputs, Weights) + Biases
        return layer

    def Update(self):
        dictionary = {}

        dictionary.update({"Weights":self.Weights})
        dictionary.update({"Biases":self.Biases})
        dictionary.update({"Weight M":self.Weight_Prev_M})
        dictionary.update({"Weight V":self.Weight_Prev_V})
        dictionary.update({"Biases M":self.Bias_Prev_M})
        dictionary.update({"Biases V":self.Bias_Prev_V})

        Data.JSON_Dump(dictionary,self.Directory)

    def Calculate_New(self):
        def Calculate_dErrors():
            errors = [] 
            
            reverse_weights = self.Weights[::-1]

            first_layer_errors = ((self.Nuerons[-1] - self.Target) * self.Nuerons[-1] * (1-self.Nuerons[-1])).tolist()[0]
            errors.append(first_layer_errors)

            for layer_weights in reverse_weights:
                layer_errors = []
                weight_sums = []
                weight_sums = [np.sum(o) for o in layer_weights.T]
                
                for o in layer_weights.T:
                    weight_sums.append(np.sum(o))

                for neuron_weights in layer_weights:
                    neuron_error = 0
            
                    for weight, error, weight_sum in zip(neuron_weights,errors[-1],weight_sums):
                        neuron_error += (weight/weight_sum) * error

                    layer_errors.append(neuron_error)
                    
                errors.append(layer_errors)
                
            errors.pop()
            errors.reverse()

            new_errors = []
            for error in errors:
                new_errors.append(np.array(error))
    
            return new_errors
        
        def New_Weight( Old_Weight, Error, Hidden_Layer, Prev_V, Prev_M):
            Error = Error * Hidden_Layer
            m = self.B1 * Prev_M + (1-self.B1) * Error
            v = self.B2 * Prev_V + (1-self.B2) * Error **2
            e = 1 ** -8

            hm = m/(1-self.B1)
            hv = v/(1-self.B2)

            new_W = Old_Weight - (hm * self.A)/ (hv**0.5 + e)
            
            return new_W, m, v

        def New_Bias( Old_Bias, Error, Prev_V, Prev_M):
        
            m = (self.B1 * Prev_M + (1-self.B1) * Error)
            v = (self.B2 * Prev_V + (1-self.B2) * Error **2)
            e = 1 ** -8
            
            hm = m/(1-self.B1)
            hv = v/(1-self.B2)

            new_B = (Old_Bias - (hm * self.A)/ (hv**0.5 + e))
            new_V = v
            new_M = m
            return new_B, new_M, new_V

        def Calculate_New_Weights():
            # Retrieve Variables or Nessary Material
            weights = self.Weights[::-1]
            nuerons = self.Nuerons[::-1]
            errors = self.Errors[::-1]
            prev_wv = self.Weight_Prev_V[::-1]
            prev_wm = self.Weight_Prev_M[::-1]

            new_weights = []
            new_wm = []
            new_wv = []

            for layer_weights, layer_wv, layer_wm, layer_errors, x in zip(weights,prev_wv, prev_wm, errors, range(self.Number_Layers)):
                new_layer_weights = []
                new_layer_wv = []
                new_layer_wm = []

                for nueron_weights, nueron_wv, nueron_wm, hidden_layer in zip(layer_weights,layer_wv,layer_wm, nuerons[x+1][0]):
                    new_neuron_weights = []
                    neuron_wv = []
                    neuron_wm = []

                    for weight, wv, wm, w_error in zip(nueron_weights,nueron_wv,nueron_wm,layer_errors):

                        new_weight, n_wm, n_wv = New_Weight(weight,w_error,hidden_layer,wv,wm)

                        neuron_wv.append(n_wv.tolist())
                        neuron_wm.append(n_wm.tolist())
                            
                        new_neuron_weights.append(new_weight)
                    

                    new_layer_wv.append(neuron_wv)
                    new_layer_wm.append(neuron_wm)
                    new_layer_weights.append(new_neuron_weights)

                new_weights.append(new_layer_weights)
                new_wv.append(new_layer_wv)
                new_wm.append(new_layer_wm)

            return new_weights[::-1], new_wm[::-1], new_wv[::-1]

        def Calculate_New_Biases():
            # Retrieve Variables or Nessary Material
            biases = self.Biases[::-1]
            errors = self.Errors[::-1]
            prev_bv = self.Bias_Prev_V[::-1]
            prev_bm = self.Bias_Prev_M[::-1]

            new_biases = []
            new_bm = []
            new_bv = []

            for layer_biases, layer_bv, layer_bm, layer_errors in zip(biases,prev_bv,prev_bm,errors):
                new_layer_biases = []
                new_layer_bm = []
                new_layer_bv = []

                for bias, bv, bm, b_error in zip(layer_biases,layer_bv,layer_bm,layer_errors):
                    
                    new_bias, n_bm, n_bv = New_Bias(bias,b_error,bv,bm)
                    
                    new_layer_biases.append(new_bias)
                    new_layer_bv.append(n_bv)
                    new_layer_bm.append(n_bm)

                new_biases.append(new_layer_biases)
                new_bv.append(new_layer_bv)
                new_bm.append(new_layer_bm)

            
            return new_biases[::-1], new_bm[::-1], new_bv[::-1]

        # Intialize Variables
        self.Errors = Calculate_dErrors()

        self.Weights,self.Weight_Prev_M,self.Weight_Prev_V = Calculate_New_Weights()

        self.Biases,self.Bias_Prev_M,self.Bias_Prev_V = Calculate_New_Biases()
    
    def Forward_Propagation(self):
        for x in range(self.Number_Layers):
            # Layer
            hidden_layer = self.Calculate_Layer(self.Nuerons[-1],self.Weights[x],self.Biases[x])
            
            if x == (self.Number_Layers-1):
                #Sigmoid Activation
                
                hidden_layer = self.Sigmoid_Activation(hidden_layer)

                num_max = max(hidden_layer.tolist()[0])
                
                output = []
                for y in hidden_layer[0]:
                    if num_max == y:
                        output.append(1)
                    else:
                        output.append(0)

                self.Outputs = output

            else:
                #Sigmoid Activation
                hidden_layer = self.Sigmoid_Activation(hidden_layer)

                # ReLU Activation
                #hidden_layer = self.ReLU_Activation(hidden_layer)
                

            self.Nuerons.append(hidden_layer)

    def Back_Propagation(self):
        self.Calculate_New()
        self.Update()     

    def Learning(self):
        self.Forward_Propagation()
        self.Back_Propagation()

    def Anwser(self):
        self.Forward_Propagation()


