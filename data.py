import torch

class data015_480_small:
    def __init__(self, logger=None, test=False, small_size=False):
        if logger is None: 
            self.print = print
        else: 
            self.print = logger.write
        
        if not test:
            self.path = '../data015/'
            self.va = Dset(self.path, 'va480_small_img.torch', 'va480_small_gts.torch', 'va480_small_gtl.torch', 'va480_small_covered_point.torch','va480_small_covered_link.torch', 'va480_small_label.torch', logger=logger)
            self.te = Dset(self.path, 'te480_small_img.torch', 'te480_small_gts.torch', 'te480_small_gtl.torch', 'te480_small_covered_point.torch','te480_small_covered_link.torch', 'te480_small_label.torch', logger=logger)
            self.tr = Dset(self.path, 'tr480_small_img.torch', 'tr480_small_gts.torch', 'tr480_small_gtl.torch', 'tr480_small_covered_point.torch','tr480_small_covered_link.torch', 'tr480_small_label.torch', logger=logger)
            self.print('loaded data480')
        else:
            self.path = '../data015/'
            self.va = Dset(self.path, 'va480_small_img.torch', 'va480_small_gts.torch', 'va480_small_gtl.torch', 'va480_small_covered_point.torch','va480_small_covered_link.torch', 'va480_small_label.torch', logger=logger)
            self.te = self.va
            self.tr = self.va
            self.print('loaded data480')
            
        if small_size:
            self.va.get_small()
            self.te.get_small()
            self.tr.get_small()
            self.print('get_small tr:va:te = %d:%d:%d'%(len(self.tr.img), len(self.va.img), len(self.te.img)))
            
class data015_480_bigger:
    def __init__(self, logger=None, test=False, small_size=False):
        if logger is None: 
            self.print = print
        else: 
            self.print = logger.write
        
        self.path = '../data015/'
        if not test:
            self.va = Dset(self.path, 'va480_bigger_img.torch', 'va480_bigger_gts.torch', 'va480_bigger_gtl.torch', 'va480_bigger_covered_point.torch', 'va480_bigger_covered_link.torch', 'va480_bigger_label.torch', logger=logger)
            self.te = Dset(self.path, 'te480_bigger_img.torch', 'te480_bigger_gts.torch', 'te480_bigger_gtl.torch', 'te480_bigger_covered_point.torch', 'te480_bigger_covered_link.torch', 'te480_bigger_label.torch', logger=logger)
            self.tr = Dset(self.path, 'tr480_bigger_img.torch', 'tr480_bigger_gts.torch', 'tr480_bigger_gtl.torch', 'tr480_bigger_covered_point.torch', 'tr480_bigger_covered_link.torch', 'tr480_bigger_label.torch',logger=logger)
        else:
            self.va = Dset(self.path, 'va480_bigger_img.torch', 'va480_bigger_gts.torch', 'va480_bigger_gtl.torch', 'va480_bigger_covered_point.torch', 'va480_bigger_covered_link.torch', 'va480_bigger_label.torch', logger=logger)
            self.te = self.va
            self.tr = self.va
        self.print('loaded data480')
        
        if small_size:
            self.va.get_small()
            self.te.get_small()
            self.tr.get_small()
            self.print('get_small tr:va:te = %d:%d:%d'%(len(self.tr.img), len(self.va.img), len(self.te.img)))
               
class Dset:
    def __init__(self, path, img, gts, gtl, covered_point, covered_link, label, logger=None):
        if logger is None: 
            self.print = print
        else: 
            self.print = logger.write
            
        self.print('loading...', path + img)
        self.img = torch.load(path + img)
        self.print('loading...', path + gts)
        self.gts = torch.load(path + gts)
        self.print('loading...', path + gtl)
        self.gtl = torch.load(path + gtl)
        
        self.print('loading...', path + covered_point)
        self.covered_point = torch.load(path + covered_point)
        self.print('loading...', path + covered_link)
        self.covered_link = torch.load(path + covered_link)
        self.print('loading...', path + label)
        self.label = torch.load(path + label)
    def get_small(self, n=None):
        if n is None:
            n = 100
        self.img = self.img[:n]
        self.gts = self.gts[:n]
        self.gtl = self.gtl[:n]
        self.covered_point = self.covered_point[:n]
        self.covered_link = self.covered_link[:n]
        self.label = self.label[:n]
        
        