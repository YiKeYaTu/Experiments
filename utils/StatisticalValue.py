class StatisticalValue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = 0
        self.min = -1
        self.count = 0
        self.values = []

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        self.values.append(self.val)

        if val > self.max:
            self.max = val

        if self.min == -1 or val < self.min:
            self.min = val