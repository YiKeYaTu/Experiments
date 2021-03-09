class StatisticalValue(object):
    """Computes and stores the average and current value"""

    def __init__(self, record=True):
        self.reset()
        self.record = record

    def reset(self):
        self.val = (None, None)
        self.max = (None, None)
        self.min = (None, None)
        self.mid = (None, None)

        self.avg = 0
        self.sum = 0

        self.count = 0
        self.values = []

    def update(self, val, identification=None, sort=False):
        self.val = (val, identification)

        self.count += 1
        
        self.sum += self.val[0]
        self.avg = self.sum / self.count

        if self.max[0] is None or self.val[0] > self.max[0]:
            self.max = self.val

        if self.min[0] is None or self.val[0] < self.min[0]:
            self.min = self.val

        if self.record:
            self.values.append(self.val)

            if sort is True:
                self.values = sorted(self.values, key=lambda x: x[0])
                self.mid = self.values[int(self.count / 2)]