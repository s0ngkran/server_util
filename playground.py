import os 
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
# img, gts, gtl, label = torch.load('4226')
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()
# gts = gts.max(0)[0].transpose(0,1)
# plt.imshow(gts)
# plt.show()
# gtl = gtl.mean(0).transpose(0,1)
# plt.imshow(gtl)
# plt.colorbar()
# plt.show()
# for _,__,fname in os.walk('va/'):
#     print('hi')

# for i, name in enumerate(fname):
#     if int(name) > 3976:
#         os.remove('va/'+name)
#         print('remove',i , name)
#     else:
#         print('hi')

os.chdir('data015/')
# img, gts, gtl, label = torch.load('110')
# np.save('temp_img_np', [img, img, img, img, img])
# dat = []
# for i in range(1000):
#     dat.append(torch.tensor(img))
# dat = torch.stack(dat)
# torch.save(dat, 'temp_img_torch')
# print(type(img))

path = 'temp/temp_2_'
img = torch.load(path + 'img')

gts = torch.load(path + 'gts')
gtl = torch.load(path + 'gtl')

plt.imshow(img)
plt.show()

plt.imshow(gts.type(torch.float32).max(0)[0].transpose(0,1))
plt.show()

plt.imshow(gtl.type(torch.float32).mean(0).transpose(0,1))
plt.show()
