
class Time:
    def __init__(self):
        import datetime

        #Finding All Aspects of Today

        today = datetime.datetime.now()

        # Creation of the Dictionary of Percents
        self.long_weekday_name = today.strftime("%A")
        self.short_weekday_name = today.strftime("%a")
        self.number_day = today.strftime("%d")

        self.number_month = today.strftime("%m")
        self.short_month_name = today.strftime("%b")
        self.long_month_name = today.strftime("%B")
        
        self.long_year = today.strftime("%Y")
        self.short_year = today.strftime("%y")

        self.today_historical_data = str(self.long_year) + "-" + str(self.number_month) + "-" + str(self.number_day) 

        self.year_historical_data = str(int(self.long_year) - 1) + "-" + str(self.number_month) + "-" + str(self.number_day) 

        self.today_data_date = self.long_year + "-" + self.number_month + "-" + self.number_day

class Data:
    def JSON_Data(Directory:str):
        import json
        file = open(Directory, "r")
        data = json.load(file)
        file.close()

        return data
                
    def JSON_Dump(Data:dict,Directory:str):
        import json
        out_file = open(Directory, "w")
        json.dump(Data, out_file, indent = 6)
        out_file.close()
                 
    def Convert_Array(list):
        import numpy as np
        array_list = []

        for obj in list:
            array_list.append(np.array(obj))

        return array_list

    def Convert_List(array_list):
        import numpy as np
        list = []

        for obj in array_list:
            list.append(obj.tolist())

        return list

    




