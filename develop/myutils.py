class ResultData:

    def __init__(self):
        self.xmin= []
        self.xmax = []
        self.ymin = []
        self.ymax = []

    def collect_bb(self,xmin,xmax,ymin,ymax):
        self.xmin.append(xmin)
        self.xmax.append(xmax)
        self.ymax.append(ymin)
        self.ymin.append(ymax)
    
    def output_bb(self):
        return self.xmin,self.xmax,self.ymin,self.ymax
    
    def is_detection(self):
        if len(self.xmin)>0:
            return True
        else:
            return False