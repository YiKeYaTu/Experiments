class StatisticalValue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = (None, None)
        self.max = (None, None)
        self.min = (None, None)

        self.avg = 0
        self.sum = 0

        self.count = 0
        self.values = []

    def update(self, val, identification=None):
        self.val = (val, identification)

        self.count += 1
        
        self.sum += self.val[0]
        self.avg = self.sum / self.count
        self.values.append(self.val)

        if self.max[0] is None or self.val[0] > self.max[0]:
            self.max = self.val

        if self.min[0] is None or self.val[0] < self.min[0]:
            self.min = self.val