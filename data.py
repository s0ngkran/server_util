import torch

class Data480_train0028:
    def __init__(self):
        self.path = '../data015/'
        self.va = Dset(self.path, 'va480_img.torch', 'va480_gts.torch', 'va480_gtl.torch', 'va480_covered.torch')
        self.te = Dset(self.path, 'te480_img.torch', 'te480_gts.torch', 'te480_gtl.torch', 'te480_covered.torch')
        self.tr = Dset(self.path, 'tr480_img.torch', 'tr480_gts.torch', 'tr480_gtl.torch', 'tr480_covered.torch')
        print('loaded data480')
        
class Dset:
    def __init__(self, path, img, gts, gtl, covered_point):
        print('loading...', img)
        self.img = torch.load(path + img)
        print('loading...', gts)
        self.gts = torch.load(path + gts)
        print('loading...', gtl)
        self.gtl = torch.load(path + gtl)
        print('loading...', covered_point)
        self.covered_point = torch.load(path + covered_point)