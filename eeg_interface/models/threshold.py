class Threshold:
    def __init__(self,threshold):
        self.threshold = threshold
        
    def shouldFlap(self,data):
        if data < self.threshold:
            return True
        else: 
            return False
    