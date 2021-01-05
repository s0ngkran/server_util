import torch
from torch.nn import functional as F
import os 
import pickle
def gen_gts(cx,cy,width,height,sigma):
    # cx -> center x
    # cy -> center y
    # width, height of image
    # sigma -> size of heat
    emt = torch.zeros([width,height])
    x, y = torch.where(emt==0)
    distx = (x-cx).float()
    disty = (y-cy).float()
    dist = distx**2+disty**2
    ans = torch.exp(-(dist/sigma**2).float())
    ans = ans.reshape([width,height])
    
    # example code
    # import torch
    # import matplotlib.pyplot as plt
    # a = gen_gts(200,200,400,600,10)
    # plt.imshow(a)
    # plt.colorbar()
    # plt.show()
    return ans
def try_gen_gts(imgfile, pklfile, ind):
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread(imgfile)
    img = img[:,:,0]/255
    img = torch.FloatTensor(img)
    
    # get data
    with open(pklfile, 'rb') as f:
        data = pickle.load(f)
    keypoint = data[str(ind)]['keypoint']

    
    ############################# config this #########################
    # scale = 
    
    gts = []
    for x,y in keypoint:
        width = 480
        height = 480
        sigma = 10
        gts.append(gen_gts(x,y,width, height, sigma))
    gts = torch.stack(gts)
    gts = gts.max(dim=0)[0].transpose(0,1)
    img = img*0.5 + gts*0.7
    # plt.title('scale='+str(scale))
    plt.imshow(img)
    plt.colorbar()
    plt.show()
        
def generate_gts(gt_file, dist_folder, dim1, dim2, sigma):
    assert gt_file[-6:] == '.torch'
    gt = torch.load(gt_file)
    keypoint = gt['keypoint']
    for i in range(len(keypoint)):
        if i == 0: continue

        #if i<=350: continue

        data = keypoint[i]
        gts = []
        for x,y in data:
            cx,cy = x,y
            width,height = dim1
            gts_ = gen_gts(cx,cy,width,height,sigma)
            gts.append(gts_)
        gts = torch.stack(gts)

        gt = gts
        width, height = dim2
        gt = F.interpolate(gt.unsqueeze(0), (width, height) ,mode='bicubic')
        gt = gt.squeeze()

        name = str(i).zfill(10)
        torch.save(gt, dist_folder+name)
        print(name, i,len(keypoint)-1)
        # plt.imshow(torch.max(a, dim=0)[0])
        # plt.show()
if __name__ == "__main__":
    # gt_file = 'training/gt_random_background_aug.torch'
    # dist_folder = 'training/gts/green/'
    # dim1 = (480,480)
    # dim2 = (60,60)
    # sigma = 18
    # generate_gts(gt_file, dist_folder, dim1, dim2, sigma)
    
    
    i = 1
    imgfile = 'data015/temp/'+str(i).zfill(10)+'.bmp'
    pklfile = 'data015/label.pkl'
    try_gen_gts(imgfile, pklfile=pklfile, ind=1)
    
    # data = torch.load('data015/gt480.torch')
    # print(len(data))
