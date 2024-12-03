class Threshold:
    def __init__(self,threshold):
        self.threshold = threshold
        
    def shouldFlap(self,data):
        print(data < self.threshold)
        if data < self.threshold:
            return True
        else: 
            return False
    