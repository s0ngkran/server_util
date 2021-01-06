import torch
import os 
import pickle
import gen_gts
import gen_gtl
import cv2
import copy
try:
    import matplotlib.pyplot as plt
except:pass


def run():
    tr = []
    va = []
    te = []
    packed = []
    # load lebel
    with open('pkl480.pkl', 'rb') as f:
        data = pickle.load(f)

    cnt = 0
    for key, value in data.items():
        label = copy.deepcopy(value)
        dat = value
        keypoint = dat['keypoint']
        side = dat['side']
        hand_scale = dat['hand-scale']

        # load img
        imgfolder = 'bmp_and_aug/'
        imgpath = os.path.join(imgfolder, str(key).zfill(10)+'.bmp')
        img = cv2.imread(imgpath)
        width, height = 480, 480
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        if side == 'left-index':
            img = cv2.flip(img, 1)
            label['side'] = 'left-index fliped'

        # gen gts
        gts = gen_gts.gen_25_keypoint(keypoint, width, height)

        # get gtl
        gtl = gen_gtl.gen_25_keypoint(keypoint, width, height)
        
       
        # append
        ans = [img, gts, gtl, label]
        person = label['person']
        if person in ['p038', 'p008', 'p020', 'p027']:
            torch.save(ans, 'datapack/va/'+str(key))
        elif person in ['p002', 'p007', 'p022']:
            torch.save(ans, 'datapack/te/'+str(key))
        else:
            torch.save(ans, 'datapack/tr/'+str(key))
        cnt += 1
        print(cnt)
    print('fin')
    # torch.save(tr, 'tr480.torch')
    # torch.save(va, 'va480.torch')
    # torch.save(te, 'te480.torch')
    def packing(path, savename):
        for _,__,fname in os.walk(path):
            print('--')
        pack = []
        for name in fname:
            pack.append(torch.load(os.path.join(path, name)))
        torch.save(pack, savename)
    print('packing te')
    packing('datapack/te/', 'te480.torch')
    print('packing va')
    packing('datapack/va/', 'va480.torch')
    print('packing tr')
    packing('datapack/tr/', 'tr480.torch')
    print('fin all')
if __name__ == "__main__":
    os.chdir('../data015')
    run()
    
    # a = torch.load('packed.torch')
    # img, gts, gtl, label = a[0]
    
    # cv2.imwrite('temp_img.bmp',img)
    # import numpy as np
    
    # gts = gts.max(0)[0].transpose(0,1)
    # gts = np.array(gts)*255
    # cv2.imwrite('temp_gts.bmp', gts)
    # gtl = gtl.mean(0).transpose(0,1)
    # gtl = np.array(gtl)*255
    # cv2.imwrite('temp_gtl.bmp', gtl)
    # print('fin')
    