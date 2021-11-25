class ResultData:
    def __init__(self):
        self.bboxs = {}

    def collect_bb(self, label, xmin, xmax, ymin, ymax):
        self.bboxs[label] = [xmin, xmax, ymin, ymax]

    def output_bb(self):
        return self.bboxs

    def is_detection(self):
        if len(self.bboxs) > 0:
            return True
        else:
            return False
