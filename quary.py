import os
import cv2
import webbrowser
import pickle
try:
    import matplotlib.pyplot as plt
except:
    pass

class MyHTML:
    def __init__(self, filename):
        self.filename = filename
        self.texttype = '<!DOCTYPE html>\n'
        self.title = 'mydata'
        
        # config
        self.id = ''
        self.clas = ''
        self.style = ''
        
        self.content = ''
        self.content_column = ['', '']
        self.top_content = ''
    def set_config(self):
        self.id = ''
        self.clas = ''
        self.style = ''
        self.set_config()
    def tag(self, tagname, content='', config=' '):
        assert type(tagname) == type(content) == str
        # adding = ''
        # if config:
        #     adding += ' id="%s" '%self.id
        #     adding += ' class="%s"'%self.clas
        #     adding += ' style="%s"'%self.style
            
        return '<'+tagname + ' '+config +'>' +content + '</'+tagname+'>' + '\n'
    def add(self, tagname, content='', config=' '):
        if type(content) != str:
            content = str(content)
            
        if tagname == 'img':
            self.content += '<img src="%s" %s>'%(content, config)
        else:
            self.content += self.tag(tagname, content, config)
        
        # for column render
        if tagname == 'img':
            self.content_column[0] += '<img src="%s" %s>'%(content, config)
        else:
            self.content_column[1] += self.tag(tagname, content=content, config=config)
    def row(self, content):
        assert type(content) == list
        column = ''
        for column_ in content:
            column += self.column(column_)
        return self.tag('div',config=' class="row" style="display:flex;"', content=column )
    def column(self, content):
        return self.tag('div',config=' class="column" style="width: 500px; padding:8px"', content=content)
    def render_normal(self):
        head = self.tag('head', '\n'+self.tag('title',self.title))
        body = self.tag('body', '\n'+self.content)
        html = self.tag('html', head + body)
        with open(self.filename+'.html', 'w') as f:
            f.write(self.texttype + html)
    def render(self):
        head = self.tag('head', '\n'+self.tag('title',self.title))
        row = self.row(self.content_column)
        body = self.tag('body', '\n'+self.top_content+ row)
        html = self.tag('html', head + body)
        with open(self.filename+'.html', 'w') as f:
            f.write(self.texttype + html)

def read_pkl(i):
    with open('label.pkl', 'rb') as f:
        data = pickle.load(f)
    data = data[str(i)]
    
    return data
        
def read_data_html(path, i, pkl_path='label.pkl' ,content=None):
    # read pkl
    if content is None:
        data = read(pkl_path)[str(i)]
        keypoint = data['keypoint']
        covered_point = data['covered_point']
        person = data['person']
        side = data['side']
        hand_scale = data['hand-scale']
        
    imgpath = os.path.join(path, str(i).zfill(10)+'.bmp')    
    temp = plot_keypoint(imgpath, keypoint, covered_point)
    
    html = MyHTML('test')
    html.add('img', temp, config='id="img" width=500px height=500px')
    html.add('h2', 'path= '+imgpath)
    html.add('p', 'person= '+person)
    html.add('p', 'hand-scale= '+str(hand_scale)+'px')
    html.add('p', 'side= '+side)
    html.add('p', 'keypoint= '+str(keypoint))
    html.add('p', 'covered_point= '+str(covered_point))
    
    html.add('button', 'hide', config='id="toggle"')
    script = """
        const img = document.getElementById('img');
        const toggle = document.getElementById('toggle');
        toggle.addEventListener('click', toggle_img);
        img.addEventListener('click', toggle_img);
        function toggle_img(){
            if (toggle.innerHTML == 'hide'){
                toggle.innerHTML = 'show';
                img.src = '%s';
            }else{
                toggle.innerHTML = 'hide';
                img.src = '%s';
            }
        }
    """%(imgpath, temp)
    html.add('script', script)
    html.render()
    webbrowser.open('test.html', new=2)
def plot_keypoint(imgpath, keypoint, covered_point, filename=None):
    if filename is None:
        filename = 'temp.jpg'
    
    img = cv2.imread(imgpath)
    class WX:
        RED = (0,0,255)
        GREEN = (0,255,0)
        BLUE = (255,0,0)
        pink = ( 245, 37, 234)
    wx = WX()
    links =       [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[6,7],[7,8],[1,4],[4,9],[9,10],[10,11],[11,12],[4,13],[13,14],[14,15],[15,16],[4,17],[17,18],[18,19],[19,20],[1,5],[5,9],[9,13],[13,17],[0,17],[21,22],[22,23],[23,24]]
    links_color = [wx.RED,wx.RED,wx.RED,wx.RED,wx.RED,wx.GREEN,wx.GREEN,wx.GREEN,wx.RED,wx.RED,wx.BLUE,wx.BLUE,wx.BLUE,wx.RED,wx.BLUE,wx.BLUE,wx.BLUE,wx.RED,wx.GREEN,wx.GREEN,wx.GREEN,wx.RED,wx.RED,wx.RED,wx.RED,wx.RED,wx.RED,wx.pink,wx.pink]
    
    for i, (ind1, ind2) in enumerate(links):
        cv2.line(img, keypoint[ind1], keypoint[ind2], links_color[i], 2)
    # draw point
    for i, center in enumerate(keypoint):
        # circle
        color = (25, 255,244 )
        if not covered_point[i]:
            cv2.circle(img, center, 4,(0,0,0) , -1)
            color = (0,255,0)
        cv2.circle(img, center, 3, color, -1)
        
        # text
        window_name = 'Image'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = center
        fontScale = 0.5
        color = (0, 0, 0) 
        thickness = 1
        cv2.putText(img, str(i), org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
    
        
        
    
    cv2.imwrite(filename,img)
    return filename
def distance(p1, p2):
    return ((p1[0]-p1[1] )**2 + (p2[0]-p2[1])**2)**0.5
def get_data(i):
    with open('label.pkl', 'rb') as f:
        data = pickle.load(f)
    return data[str(i)]
def write_scale():
    with open('label.pkl', 'rb') as f:
        alldata  = pickle.load(f)
    scale = []
    for i in range(1, 3977):
        data = get_data(i)
        keypoint = data['keypoint']
        _scale = (distance(keypoint[5], keypoint[17]) + distance(keypoint[9], keypoint[12]))*0.5
        data['hand-scale'] = int(_scale)
        alldata[str(i)]['hand-scale'] = int(_scale)
        print(i)
  
    with open('label_with_scale.pkl', 'wb') as f:
        pickle.dump(alldata, f)
def read(filename):
    with open(filename, 'rb') as f:
        data  = pickle.load(f)
    return data
def write_all_scale():
    scale = []
    data = read('label.pkl')
    for i in range(1, len(data)+1):
        _data = data[str(i)]
        _scale = _data['hand-scale']
        scale.append(_scale)
    
    with open('allscale.pkl', 'wb') as f:
        pickle.dump(scale, f)  
def get_scale():
    data = read('allscale.pkl')
    data.sort()
    print(data[-100:])
    # plt.plot(data,'.')
    # plt.ylabel('scale')
    # plt.xlabel('n_img')
    # plt.show()
def find_scale(scale):
    data = read('label.pkl')
    for i in range(1, len(data)):
        if data[str(i)]['hand-scale'] == scale:
            print(i, scale)
            break
    return i

if __name__ == "__main__":
    
    get_scale()
    i = find_scale(19)
    read_data_html('bmp/',i)
    
    # check aug_img in temp/ folder
    # for _,__, fname in os.walk('temp/'):
    #     print('fin')
    
    # fname.sort()
    # for namei in fname:
    #     i = int(namei[:10])
    #     read_data_html('temp', i, pkl_path='aug_label.pkl' )
    #     input('>>')