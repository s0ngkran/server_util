import os 
import torch
import cv2
import pickle
import sys
sys.path.append('..')
from util import gen_gts
from util import gen_gtl

class Pack:
    def __init__(self, n, name):
        assert type(name) == str
        size = 120
        self.name = name
        self.img = torch.empty(n, 480, 480, 3, dtype=torch.uint8) # np array
        self.gts = torch.empty(n, 25, size, size, dtype=torch.float16)
        self.gtl = torch.empty(n, 46, size, size, dtype=torch.float16)
        self.covered_point = torch.empty(n, 25, dtype=torch.bool)
        self.covered_link = torch.empty(n, 46, dtype=torch.bool)
        self.label = []
        self.cnt = 0
    def fill(self, img, gts, gtl, covered_point, covered_link, label):
        self.img[self.cnt] = torch.tensor(img, dtype=torch.uint8)
        self.gts[self.cnt] = gts
        self.gtl[self.cnt] = gtl
        self.covered_point[self.cnt] = covered_point
        self.covered_link[self.cnt] = covered_link
        self.label.append(label)
        self.cnt += 1
    def write(self):
        print('saving...')
        torch.save(self.img, self.name+'_img.torch')
        torch.save(self.gts, self.name+'_gts.torch')
        torch.save(self.gtl, self.name+'_gtl.torch')
        torch.save(self.covered_point, self.name+'_covered_point.torch')
        torch.save(self.covered_link, self.name+'_covered_link.torch')
        torch.save(self.label, self.name+'_label.torch')
        print('finished')

def run():    
    def manage_img(img, side):
        img = cv2.resize(img, (480,480))
        if side == 'left-index':
            img = cv2.flip(img, 1)
            print('found')
        return img
    def gen_covered_link(covered_point):
        link25 =  [[0,1],[1,2],[2,3],[0,4],[5,6],[6,7],[7,8],[4,9],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[21,22],[22,23],[23,24]]
        empty = torch.empty(len(link25)*2, dtype=torch.bool).fill_(False)
        covered = []
        for i, covered_ in enumerate(covered_point):
            if covered_:
                covered.append(i)
        for i, (a, b) in enumerate(link25):
            if (a in covered) or (b in covered):
                empty[i*2] = True
                empty[i*2+1] = True
        return empty
    def gen_data(key, val):
        keypoint = val['keypoint']
        width, height = 480, 480
        img = cv2.imread('bmp_and_aug/'+str(key).zfill(10)+'.bmp')
        img = manage_img(img, val['side'])
        gts = gen_gts.gen_25_keypoint(keypoint, width, height)
        gtl = gen_gtl.gen_25_keypoint(keypoint, width, height)
        covered_point = torch.tensor(val['covered_point'], dtype=torch.bool)
        covered_link = gen_covered_link(val['covered_point'])
        val['id'] = key
        return img, gts, gtl, covered_point, covered_link, val
    trset = Pack(7122, 'tr480_small')
    vaset = Pack(138, 'va480_small')
    teset = Pack(277, 'te480_small')

    with open('pkl480.pkl', 'rb') as f:
        data = pickle.load(f)

    tr, va, te, out = 0,0,0,0
    cnt = 1
    for key, val in data.items():
        person = val['person']
        
        
        if person in ['p038', 'p008', 'p020', 'p027'] :
            if int(key) <= 3976:
                va += 1
                img, gts, gtl, covered_point, covered_link, val = gen_data(key, val)
                vaset.fill(img, gts, gtl, covered_point, covered_link, val)
                pass
            else:
                out += 1
        elif person in ['p002', 'p007', 'p022']:
            if int(key) <= 3976:
                te += 1
                img, gts, gtl, covered_point, covered_link, val = gen_data(key, val)
                teset.fill(img, gts, gtl, covered_point, covered_link, val)  
                pass 
            else:
                out += 1
        else:
            tr += 1
            img, gts, gtl, covered_point, covered_link, val = gen_data(key, val)
            trset.fill(img, gts, gtl, covered_point, covered_link, val) 
            pass
        print(tr, va, te, out, tr+va+te+out, len(data))
    vaset.write()
    teset.write()
    trset.write()
    print('fin trset')


def check():
    import numpy as np
    name = 'va480_bigger'
    i =-1
    img = torch.load( name+'_img.torch')
    cv2.imwrite('temp_img.bmp',np.array(img[i]))
    gts = torch.load( name+'_gts.torch')
    cv2.imwrite('temp_gts.bmp',np.array((gts[i].type(torch.float32)*255).max(0)[0].transpose(0,1)))
    gtl = torch.load( name+'_gtl.torch')
    cv2.imwrite('temp_gtl.bmp',np.array((gtl[i].type(torch.float32)*255).mean(0).transpose(0,1)))
    covered_point = torch.load( name+'_covered_point.torch')
    print('covered_point=', covered_point[i])
    covered_link = torch.load( name+'_covered_link.torch')
    print('covered_link=', covered_link[i])
    label = torch.load(name+'_label.torch')
    print('label=', label[i])
    
if __name__ == "__main__":
    run()
    # check()
    