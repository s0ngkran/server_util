import numpy as np 
import torch 
import cv2 
import os 

def makeimgs(img_folder, savefile):
    assert img_folder[-1] == '/'
    assert savefile[-6:] == '.torch'
    print('walking')
    for _,__,img_names in os.walk(img_folder):
        print('fin walk')
    assert len(img_names)
    
    n_imgs = len(img_names)
    for i in range(n_imgs):
        assert img_names[i][-4:] == '.bmp'
    print('checked .bmp')   

    img = cv2.imread(img_folder+img_names[0]) # y,x,channel
    y,x,ch = img.shape
    assert ch == 3
    shape = (n_imgs+1, ch, x, y)
    temp = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
    print('checked empty_memory of shape',shape)
    
    for i, _ in enumerate(img_names): #!
        i += 1
        name = str(i).zfill(10) +'.bmp'
        img = cv2.imread(img_folder+name).transpose(2,1,0)
        img = torch.IntTensor(img)
        temp[i] = img
        print(i,'/',n_imgs)
    print('')
    print('writing...')
    torch.save(temp, savefile)
    print('saved %s'%savefile)
    print('-------------finish-----------------')

def test_makeimgs():
    img_folder = 'example_folder/'
    savefile = 'example_folder_save/testsave_imgs.torch'
    makeimgs(img_folder, savefile)

def makegt(gt_folder, savefile):
    assert gt_folder[-1] == '/'
    assert savefile[-6:] == '.torch'
    print('walking')
    for _,__,gt_names in os.walk(gt_folder):
        print('fin walk')

    for gt_name in gt_names:
        assert int(gt_name), 'got bad format %s'%gt_name
    print('checked gtname')

    n_gt = len(gt_names)
    temp = torch.load(gt_folder + gt_names[0])
    part, x, y = temp.shape
    shape = n_gt+1, part, x, y # [1,1,2,3,4,5,6,......]
    gt = torch.empty(shape, dtype=torch.float32, pin_memory=False)
    print('checked empty_memory of shape',shape)
    
    for i,_ in enumerate(gt_names): #!
        i+=1
        name = str(i).zfill(10)
        gt_ = torch.load(gt_folder+name)
        gt[i] = gt_ 
        print(i,'/',n_gt)
    print('')
    print('writing...')
    torch.save(gt, savefile)
    print('saved',savefile)
    
def test_makegt():
    makegt( gt_folder = 'example_folder_gts/',
            savefile = 'example_folder_save/savegts.torch')

def check_data(imgs_file, gts_file, gtl_file, inds=None,savefolder='temp/'):
    from torchvision.utils import save_image
    from torch.nn import functional as F
    assert imgs_file[-6:] == gts_file[-6:] == gtl_file[-6:] == '.torch'
    
    print('loading...')
    imgs = torch.load(imgs_file) # ind, ch, x, y
    if inds is None:
        n_imgs = len(imgs)-1
        inds = torch.randperm(n_imgs)[:5]
        inds = [i.item() for i in inds]
    assert type(inds)==list
    gts = torch.load(gts_file) # ind, part, x, y
    gtl = torch.load(gtl_file) # ind, part, x, y
    print('loaded imgs')
    size = imgs[0,0].shape
    for i in inds:
        img_ = imgs[i] # ch, x, y
        gts_ = gts[i]
        gtl_ = gtl[i]
     
        img_ = img_.to(torch.float32)/255
     
        img_ = img_.transpose(1,2)
        gts_ = gts_.max(0)[0].transpose(1,0)
        gtl_ = gtl_.mean(0).transpose(1,0)

        gts_ = F.interpolate(gts_.unsqueeze(0).unsqueeze(0), size ,mode='bicubic').squeeze().squeeze()
        gtl_ = F.interpolate(gtl_.unsqueeze(0).unsqueeze(0), size ,mode='bicubic').squeeze().squeeze()

        save_image(img_, savefolder+str(i)+'_img.bmp')
        save_image(gts_, savefolder+str(i)+'_gts.bmp')
        save_image(gtl_, savefolder+str(i)+'_gtl.bmp')
        print('saved', i)


if __name__ == "__main__":
    # folder = 'training/img/replaced_background/'
    # savefile = 'training/imgs_replaced_background.torch'
    # makeimgs(folder, savefile)

    # folder = 'training/gts/replaced_background/'
    # savefile = 'training/gts_replaced_background.torch'
    # makegt(folder, savefile)

    # a = input('continue gtl?')
    # for name in ['green_screen','random_background','replaced_background','replaced_green']:

    #     folder = 'training/gtl/'+name +'/'
    #     savefile = 'training/gtl_' +name+'.torch'
    #     makegt(folder, savefile)

    #check data
    img = 'training/imgs_replaced_samecam.torch'
    gts = 'training/gts_replaced_samecam.torch'
    gtl = 'training/gtl_replaced_samecam.torch'
    lst = [i+30 for i in range(20)]
    lst2 = [i+5000 for i in range(20)]
    lst3 = [i+13900 for i in range(20)]
    lst.extend(lst2)
    lst.extend(lst3)
    check_data(img, gts, gtl, lst)


