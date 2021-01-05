import os 
import pickle

for _,__,fname in os.walk('data015/pkl480/'):
    print('fin')

fname.sort()
data = {}
for name in fname:
    i = int(name[:10])
    print(i)
    path = 'data015/pkl480/'+name
    with open(path, 'rb') as f:
        _data = pickle.load(f)
    data[str(i)] = _data
    
    #
    
    break
    
with open('data015/pkl480.pkl', 'wb') as f:
    pickle.dump(data, f)
    
with open('data015/pkl480.pkl', 'rb') as f:
    data = pickle.load(f)
    
    print('===',data)
