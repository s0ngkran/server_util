import torch 
from torchvision.utils import save_image
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import torch.nn.functional as F
class torch2img():
    def __init__(self, savename):
        assert str(savename) == str(savename)
        self.savename = str(savename)+'_'

    def write(self, torch_img, inverse = True ,filename=None,gt=None ):
        if filename is None: filename = self.savename + 'temp'
        assert len(torch_img.shape) in [2,3]
        filename = filename + '.jpg'
        if len(torch_img.shape) == 3:
            channel = torch_img.shape[0]
            assert channel in [3,20,11,2,25,46], '[channel, x, y] need channel in [3, 20, 11] but got ' + str(torch_img.shape)
            if channel == 20:
                #gtl mode
                torch_img = torch_img.mean(0)
                mx = torch_img.max()
                mn = torch_img.min()
                torch_img = (torch_img-mn)/(mx-mn)
                mx = torch_img.max()
                mn = torch_img.min()
                mean = torch_img.mean()
                mid = (mx-mn)/2
                if mean<mid:
                    inverse = True
                else: inverse = False
            elif channel == 11:
                torch_img = torch_img.max(0)[0]


            elif (channel == 2 or channel==25) and gt == 'gts':
                torch_img = torch_img.max(0)[0]
            elif (channel == 2 or channel==46) and gt == 'gtl':
                torch_img = torch_img.mean(0)
          


        if len(torch_img.shape) == 2:
            img = torch_img.transpose(1,0) # get horizontal
        if len(torch_img.shape) == 3:
            inverse = False
            img = torch_img[(2,1,0),:,:].transpose(1,2) # get horizontal
        if inverse:
            img = 1-img
        save_image(img, filename)
    
    def vcat(self,tensors, gt):
        n = len(tensors)
        for i in range(n):
            self.write(tensors[i], filename='temp/'+self.savename+'temp%s'%i, gt=gt)
        npys = [cv2.imread('temp/'+self.savename+'temp%s.jpg'%i) for i in range(n)]
        cv2.imwrite('temp/'+self.savename+'temp.jpg',cv2.vconcat(npys))
   
    def test_write(self,):
        # t = torch.rand([3,60,60])
        # t = torch.rand([60,60])
        t = torch.load('ex_gtl')
        self.write(t)
        self.read()
    def test_vcat(self,):
        t = torch.load('ex_img')
        ts = torch.stack([t,t,t])
        self.vcat(ts)
        self.read()
    def test_hcat(self,):
        t = torch.rand([3,3,60,60])
        t = torch.load('ex_img')
        ts = [t,t,t]
        img = self.hcat(ts)
        cv2.imwrite('temp/'+self.savename+'temp.jpg',img)
        self.read()
    def read(self,):
        import matplotlib.pyplot as plt
        img = cv2.imread('temp/'+self.savename+'temp.jpg')
        plt.imshow(img)
        # plt.colorbar()
        plt.show()
    def test_cat(self,):
        import torch 
        # t1 = torch.rand([60,60])
        t1 = torch.load('ex_img')[0]
        tt = [t1,t1,t1]
        img = self.cat_x(tt)
        self.write(img, inverse=True)
        self.read()
    def resize(self,torch, size):
        assert len(torch.shape) == len(size) == 2
        import torch.nn.functional as F
        resized = F.interpolate(torch.unsqueeze(0).unsqueeze(0), size, mode='bicubic')
        return resized.squeeze().squeeze()
    def test_resize(self,):
        img = torch.load('0000000004')[0]
        print('img=',img.shape)
        gts = torch.load('0000000004_')
        gts = torch.max(gts, dim=0)[0]
        print('gts=',gts.shape)
        gts = self.resize(gts, [40,40])
        print('gts_resized=',gts.shape)

        # img = img*0.5+gts*0.5
        self.write(gts)
        self.read()
    def test_write_gtl(self,):
        img = torch.load('0000000004_')
        img = torch.mean(img, dim=0)
        img = self.resize(img, [480,480])
        self.write(img, inverse=True)
        self.read()
    def write_all(self, tensors):
        imgs = []
        for i,tensor in enumerate(tensors):
            if i==5:
                self.vcat(tensor, gt='gtl')
            else:
                self.vcat(tensor, gt='gts')
            imgs.append(cv2.imread('temp/'+self.savename+'temp.jpg'))
        img = cv2.hconcat(imgs)
        # cv2.imwrite(self.savename+'temp.jpg',img)
        return img
    def test_write_all(self,):
        t = torch.load('ex_img')
        ts = torch.stack([t, t, t])
        tss = [ts,ts,ts]
        img = self.write_all(tss)
        cv2.imwrite('temp/'+self.savename+'temp.jpg', img)
        self.read()

    def feed(self,):
        from hand_model import hand_model
        import torch
        import torch.nn.functional as F
        # model = hand_model()
        img = torch.load('ex_img')
        img = torch.stack([img,img])
        # out = model(img)

        gts = torch.load('ex_gts')
        gtl = torch.load('ex_gtl')
        gts = torch.stack([gts, gts])
        gtl = torch.stack([gtl, gtl])
        # gts_pred = out[11]
        # gtl_pred = out[5]
        
        print(img.shape, gts.shape)
        # print(img[:,0,:,:].shape, gts.max(0)[0].shape)
        img = F.interpolate(img, [60,60])
        img_ = img[:,0,:,:]*0.4 + gts.max(1)[0]*0.6
        # print(img.shape, gts.shape, gtl.shape, gts_pred.shape, gtl_pred.shape)
        self.write_all([img,img_,gts,gtl])
        # write_all([img, gts, gts_pred, gtl, gtl_pred])
        self.read()
    def genimg(self, img, out, gts, gtl):
        assert gts[0,0,0] != 60, 'solve this assert'
        img = F.interpolate(img, gts[0,0].shape)
        gts_pred = out[11]
        gtl_pred = out[5]
        img_ = img[:,0,:,:]*0.4 + gts_pred.max(1)[0]*0.6
        self.write_all([img,img_,gts,gts_pred,gtl,gtl_pred])
    def genimg_(self, img, out, gts, gtl, filename, msg='temp', savefolder=None): #img = [batch, ch, x, y]
        assert gts[0,0,0].shape != 45, 'solve this assert'
        # print('gtl in ',type(gtl))
        assert len(gts.shape) == len(gtl.shape) == 4
        assert len(out) == 2 # predL, predS
        # print('---',out[0].shape, out[1].shape, gtl.shape)
        assert out[0].shape == gtl.shape
        assert out[1].shape == gts.shape
        if savefolder is None: savefolder = 'saveimg/'
        img = F.interpolate(img, gts[0,0].shape)
        gtl_pred = out[0]
        gts_pred = out[1]
        img_ = img[:,0,:,:]*0.4 + gts_pred.max(1)[0]*0.6
        if img.shape[1] == 1:
            img = torch.cat([img, img, img], dim=1)
        img = self.write_all([img, img_, gts, gts_pred, gtl, gtl_pred])
        img = self.write_header_msg(img, msg)
        tempfolder = 'temp/'
        cv2.imwrite('temp/'+self.savename+'temp.jpg',img) # for send to line
        cv2.imwrite(savefolder + self.savename+filename+'.jpg', img)
    def write_header_msg(self, oriimg, msg):
        assert len(oriimg.shape) == 3
        assert type(msg)==str, 'msg argument needs str type'
        img = np.zeros((35, oriimg.shape[1], 3))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (10, 22) 
        fontScale = 0.6
        color = (255, 255, 255)
        thickness = 2
        img = cv2.putText(img, msg, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
        img = np.vstack((img, oriimg))
        cv2.imwrite('temp/'+self.savename+'testhead.jpg', img)
        return img
        # cv2.imwrite(self.savename+'header.jpg',img)
    def test_write_header_msg(self,):
        tempimg = cv2.imread('temp/'+self.savename+'temp.jpg')
        self.write_header_msg(tempimg, 'tr te va epoch')
        print('open %s to check'%(self.savename+'temp.jpg'))
if __name__ == '__main__':
    test = torch2img('test')
    test.test_write_header_msg()
    pass
