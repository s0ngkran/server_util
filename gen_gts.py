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
def distance (p1, p2):
    distx = (p1[0]-p2[0])**2
    disty = (p1[1]-p2[1])**2
    return (distx+disty)**0.5
def gen_25_keypoint(keypoint, width, height):
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    big_link = [[4,5],[4,9],[4,13],[4,17],[4,0]]
    dist_big = [distance(keypoint[i], keypoint[j]) for i,j in big_link]
    dist_big = sum(dist_big)/len(dist_big)
    
    small_sigma = dist_finger*0.4
    big_sigma = dist_big*0.4
    gts = []
    
    ### config ####
    for ind, (x,y) in enumerate(keypoint):
        if ind in [0,1,4,21]:
            sigma = big_sigma
        else:
            sigma = small_sigma
        gts.append(gen_gts(x,y,width, height, sigma))
    gts = torch.stack(gts)
    gts = F.interpolate(gts.unsqueeze(0), size=(120,120), mode='bicubic').squeeze(0)
    return gts
def try_gen_gts(imgfile, pklfile, ind):
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread(imgfile)
    img = img[:,:,0]/255
    img = torch.FloatTensor(img)
    
    # get data
    with open(pklfile, 'rb') as f:
        data = pickle.load(f)
    data = data[str(i)]
    keypoint = data['keypoint']
    
    ############################# config this #########################
    hand_scale = data['hand-scale']
    
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    big_link = [[4,5],[4,9],[4,13],[4,17],[4,0]]
    dist_big = [distance(keypoint[i], keypoint[j]) for i,j in big_link]
    dist_big = sum(dist_big)/len(dist_big)
    
    small_sigma = dist_finger*0.4
    big_sigma = dist_big*0.4
    
    # print(hand_scale, sigma, dist_finger)
    
    gts = []
    width = 480
    height = 480
    for ind, (x,y) in enumerate(keypoint):
        if ind in [0,1,4,21]:
            sigma = big_sigma
        else:
            sigma = small_sigma
        gts.append(gen_gts(x,y,width, height, sigma))
    gts = torch.stack(gts)
    gts = F.interpolate(gts.unsqueeze(0), size=(120,120), mode='bicubic').squeeze(0)
    gts = gts.max(dim=0)[0].transpose(0,1)
    # img = img*0.5 + gts*0.7
    # plt.title('scale='+str(scale))
    
    print(gts.shape)
    plt.imshow(gts)
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
    
    
    i = 7621
    imgfile = 'data015/temp/'+str(i).zfill(10)+'.bmp'
    pklfile = 'data015/pkl480.pkl'
    try_gen_gts(imgfile, pklfile=pklfile, ind=i)
    
    # data = torch.load('data015/gt480.torch')
    # print(len(data))
