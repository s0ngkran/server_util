import torch
import numpy as np
import pickle 
import os
from torch.nn import functional as F
link = [[0,1] ,[0,3] ,[0,5] ,[0,7] ,[0,9], [1,2], [3,4], [5,6], [7,8], [9,10]]
link = [[0,1]]
def gt_vec(width, height, point, size=20, link=link):
    ans = torch.zeros( len(link)*2, width, height)
    x, y = np.where(ans[0]==0)
    for index, (partA, partB) in enumerate(link):
        vec = point[partB]-point[partA]
        length = np.sqrt(vec[0]**2+vec[1]**2)
        u_vec = vec/length
        u_vec_p = np.array([u_vec[1], -u_vec[0]])

        tempx = x-point[partA][0]
        tempy = y-point[partA][1]
        temp_ = []
        temp_.append(tempx)
        temp_.append(tempy)
        temp = np.stack(temp_)
  
        c1 = np.dot(u_vec,temp)
        c1 = (0<=c1) & (c1<=length)
        c2 = abs(np.dot(u_vec_p,temp)) <= size
        condition = c1 & c2
       
        ans[ index*2] = torch.tensor(u_vec[0] * condition).reshape(width, height)  #x
        ans[ index*2+1] = torch.tensor(u_vec[1] * condition).reshape(width, height) #y
    return ans
def distance (p1, p2):
    distx = (p1[0]-p2[0])**2
    disty = (p1[1]-p2[1])**2
    return (distx+disty)**0.5
def gen_25_keypoint(keypoint, width, height):
    link25 =  [[0,1],[1,2],[2,3],[0,4],[5,6],[6,7],[7,8],[4,9],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[21,22],[22,23],[23,24]]
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    small_sigma = dist_finger*0.4
    
    keypoint = [np.array(i) for i in keypoint]
    gtl = gt_vec(width, height, keypoint,size=small_sigma, link=link25)
    
    
    gtl = F.interpolate(gtl.unsqueeze(0), size=(120,120), mode='nearest').squeeze(0)
    return gtl
    
def test_gt_vec():
    import matplotlib.pyplot as plt 
    link25 =  [[0,1],[1,2],[2,3],[0,4],[5,6],[6,7],[7,8],[4,9],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[21,22],[22,23],[23,24]]
    i = 1
    
    with open('pkl480.pkl', 'rb') as f:
        data = pickle.load(f)
    dat = data[str(i)]
    point = dat['keypoint']
    keypoint = point
    
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    small_sigma = dist_finger*0.4
    
    point = [np.array(i) for i in point]
    gtl = gt_vec(480,480, point,size=small_sigma, link=link25)
    
    
    gtl = F.interpolate(gtl.unsqueeze(0), size=(120,120), mode='nearest').squeeze(0)
    print(gtl.shape)
    gtl = gtl.mean(0)
    
    thres = 120/480 
    for x,y in point:
        x, y = x*thres, y*thres
        plt.plot(x,y,'r.')
        
    plt.imshow(gtl.transpose(0,1))
    plt.show()
def gen_gtl_folder(gt_file, savefolder, dim1, dim2, size):
    assert gt_file[-6:] == '.torch'
    assert savefolder[-1] == '/'
    from torch.nn import functional as F

    gt = torch.load(gt_file)
    keypoint = gt['keypoint']
    for i in range(len(keypoint)):
        if i == 0: continue
        
        # if i<=350: continue
        point = [np.array(x) for x in keypoint[i]]
        width, height = dim1
        gtl = gt_vec(width,height,point,size=size)
        
        gtl = F.interpolate(gtl.unsqueeze(0), dim2, mode='nearest').squeeze()
        # import matplotlib.pyplot as plt
        # plt.imshow(gtl[0])
        # plt.show()
        name = str(i).zfill(10)
        torch.save(gtl, savefolder+name)
        print(name, i, len(keypoint))
    print('fin all')
def test_gtl():
    import matplotlib.pyplot as plt 
    import cv2
    from torch.nn import functional as F
    img = cv2.imread('example_folder/0000000010.bmp') # y,x,ch
    print(img.shape)
    img = torch.FloatTensor(img/255).transpose(0,2) # ch,x,y

    gtl = torch.load('example_folder_gtl/0000000010')
    
    gtl = F.interpolate(gtl.unsqueeze(0), (360,360), mode='nearest').squeeze()
    # gtl = gtl.mean(0)
    # ans = img[0]*0.01+ gtl*0.5
    ans = gtl[0]
    plt.imshow(ans)
    plt.colorbar()
    plt.show()
def ex_gen_gtl_folder():
    gt_folder = 'testing/pkl/'
    savefolder = 'testing/gtl/'
    dim1 = (360,360)
    dim2 = (45,45)
    gen_gtl_folder(gt_folder, savefolder, dim1, dim2)

if __name__ == "__main__":
    # lst = ['gt_random_background_aug.torch','gt_replaced_background.torch','gt_replaced_green.torch']
    # lst2 = ['random_background','replaced_background','replaced_green']
    # a = zip(lst, lst2)
    # for gt, savefol in a:
    #     gt_file = 'training/'+gt
    #     savefolder = 'training/gtl/'+savefol+'/'
    #     dim1 = (360,360)
    #     dim2 = (45,45)
    #     size = 10
    #     gen_gtl_folder(gt_file, savefolder, dim1, dim2, size)
    #     print('-----------------------------')
   
    os.chdir('data015')
    test_gt_vec()