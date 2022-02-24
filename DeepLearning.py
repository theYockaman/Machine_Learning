import numpy as np
from Functions import Data

class DeepLearning:
    def __init__(self,Inputs:list,Number_Outputs:int , Target:list = None, Directory:str = "/Users/nathanyockey/Desktop/Financial Engineering/Data/Stocks/StockLearning.json"):
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

                list_biases = np.ones((1, final))
            
                biases.append(list_biases)

        return biases
    
    def Intialize_Weight_M(self):
        file = Data.JSON_Data(self.Directory)

        if file.get("Weights M"):
            m_s = Data.Convert_Array(file['Weights M'])
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

        if file.get("Weights V"):
            v_s = Data.Convert_Array(file['Weights V'])
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

        if file.get("Biased M"):
            biases = Data.Convert_Array(file['Biases M'])
        else:
            biases = []

            for layer in range(self.Number_Layers):

                if layer == (self.Number_Layers-1):
                    final = self.Number_Outputs

                else:
                    final = self.NueronsPerLayer

                list_biases = np.zeros((1,final))
                
                biases.append(list_biases)

        return biases

    def Intialize_Bias_V(self):
        file = Data.JSON_Data(self.Directory)

        if file.get("Biases V"):
            v_s = Data.Convert_Array(file['Biases V'])
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
        import Programming.Functions as Functions
        dictionary = {}

        dictionary.update({"Weights":Data.Convert_List(self.Weights)})
        dictionary.update({"Biases":Data.Convert_List(self.Biases)})
        dictionary.update({"Weight M":Data.Convert_List(self.Weight_Prev_M)})
        dictionary.update({"Weight V":Data.Convert_List(self.Weight_Prev_V)})
        dictionary.update({"Biases M":Data.Convert_List(self.Bias_Prev_M)})
        dictionary.update({"Biases V":Data.Convert_List(self.Bias_Prev_V)})

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
        
            m = self.B1 * Prev_M + (1-self.B1) * Error
            v = self.B2 * Prev_V + (1-self.B2) * Error **2
            e = 1 ** -8
            
            hm = m/(1-self.B1)
            hv = v/(1-self.B2)

            new_B = Old_Bias - (hm * self.A)/ (hv**0.5 + e) 
            
            return new_B, m, v

        self.Errors =  Calculate_dErrors() 
        new_biases = []
        new_weights = []

        new_bm = []
        new_bv = []

        new_wm = []
        new_wv = []
        
        # Reverse Biases, Weights, Errors, and Nuerons
        reverse_biases = self.Biases[::-1]
        reverse_weights = self.Weights[::-1]
        reverse_errors = self.Errors[::-1]
        reverse_nuerons = self.Nuerons[::-1]
        reverse_bv = self.Bias_Prev_V[::-1]
        reverse_bm = self.Bias_Prev_M[::-1]
        reverse_wv = self.Weight_Prev_V[::-1]
        reverse_wm = self.Weight_Prev_M[::-1]

        # Calculate New Biases
        for x in range(self.Number_Layers):
            layer_biases = []
            layer_v = []
            layer_m = []
            
            for bias, error, v, m, y in zip(reverse_biases[x],reverse_errors[x], reverse_bv[x], reverse_bm[x], range(len(reverse_errors[x]))):

                new_bias, new_m, new_v = New_Bias(bias,error,v,m)

                layer_v.append(new_v)
                layer_m.append(new_m)
                
                layer_biases.append(new_bias)

            new_biases.append(np.array(layer_biases))
            new_bv.append(np.array(layer_v))
            new_bm.append(np.array(layer_m))

        # Calculate New Weights
        for layer_weights, x in zip(reverse_weights,range(self.Number_Layers)):
            layer_new_weights = []
            layer_v = []
            layer_m = []
               
            for neuron_weights, neuron_wv, neuron_wm, hl in zip(layer_weights,reverse_wv[x], reverse_wm[x], reverse_nuerons[x+1][0]):
                new_neuron_weights = []
                neuron_v = []
                neuron_m = []
                
                for weight, error, v, m in zip(neuron_weights,reverse_errors[x],neuron_wv, neuron_wm):
                    
                    new_weight, new_m, new_v = New_Weight(weight,error,hl,v,m)

                    neuron_v.append(new_v)
                    neuron_m.append(new_m)
                    
                    new_neuron_weights.append(new_weight)

                layer_v.append(neuron_v)
                layer_m.append(neuron_m)
                layer_new_weights.append(new_neuron_weights)

            new_weights.append(np.array(layer_new_weights))
            new_wv.append(np.array(layer_v))
            new_wm.append(np.array(layer_m))

        # Reversing the New Weights & Biases
        self.Weights = new_weights[::-1]
        self.Biases = new_biases[::-1]
        self.Bias_Prev_M = new_bm[::-1]
        self.Bias_Prev_V = new_bv[::-1]
        self.Weight_Prev_M = new_wm[::-1]
        self.Weight_Prev_V = new_wv[::-1]
                  
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

