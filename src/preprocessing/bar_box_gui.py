from logging.handlers import RotatingFileHandler
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QAction, QComboBox, QLabel, QInputDialog
from PyQt5.QtGui import QColor, QBrush, QPainter, QPixmap, QPolygonF, QPen
from PyQt5.QtCore import QPoint, QRect, QPointF, QThread, QTimeLine, Qt
import matplotlib.pyplot as plt
from sympy import re
import numpy as np
import os
import json
from scipy import ndimage
# GUI for identifying 
class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Microstructure Inpainter')
        self.painter_widget = PainterWidget(self) 
        self.setCentralWidget(self.painter_widget)
        self.setGeometry(30, 30, self.painter_widget.image.width(), self.painter_widget.image.height())
        self.show()

    def keyPressEvent(self, event):
        self.painter_widget.keyPressEvent(event)

class PainterWidget(QWidget):
    def __init__(self, parent):
        super(PainterWidget, self).__init__(parent)
        self.parent = parent
        screen = self.parent.app.primaryScreen()
        self.screensize = screen.size()
        self.data_path = 'data/micrographs_raw'
        self.img_list = sorted(os.listdir(self.data_path))
        with open(f'data/good_micros.json', 'r') as f:
                self.good = json.load(f)
        try:
            with open(f'data/anns.json', 'r') as f:
                    self.data_map = json.load(f)
                    print('loading')
        except:
            self.data_map = {}
        self.current_img = 0
        # while str(self.current_img) in self.data_map.keys():
        #     self.current_img+=1
        #     self.img_name = self.img_list[self.good[self.current_img]]
        #     self.img_path = f'{self.data_path}/{self.img_name}'
        #     self.img = plt.imread(self.img_path)
        
        self.img_name = self.img_list[self.current_img]
        self.img_path = f'{self.data_path}/{self.img_name}'
        self.img = plt.imread(self.img_path)
        self.setPixmap(self.img_path, loading=True)
        self.boundary=0.01
        self.phases = []
        self.selected_phase = 1
        self.opaque = 1
        self.rot=0
        self.stage = QLabel('scale bar col: click on the scale bar then use A and S keys to adjust thresholds, or press enter to skip')
        self.modes = ['scale bar col: click on the scale bar then use A and S keys to adjust thresholds, or press enter to skip', 
                        'scale bar box: click on the top left then bottom right corners of the reqion containing the scale bar, or press enter to skip', 
                        'crop region: click on the top left then bottom right corner to define the region you want to keep. Click a third time to reset. Press enter to skip',  
                        'Click on the different phases to segment. Use A and S to adjust threshold',
                        'rotate: use A and S to rotate',
                        'voxel size: click on the left of the scalebar, then the right, then enter scale bar size in microns',
                        'scale bar col: click on the scale bar then use A and S keys to adjust thresholds, or press enter to skip', ]
        label = parent.addToolBar('stage')
        label.addWidget(self.stage)
        self.parent.addToolBarBreak()
        self.general = QLabel('At any time, press C to restart the microstructure, or W to remove the microstructure')
        label = parent.addToolBar('general')
        label.addWidget(self.general)
        self.clear()
        self.show()


    def setPixmap(self, fp, loading=True):
        self.image = QPixmap(fp)
        if loading:
            x, y = self.screensize.width(), self.screensize.height()
            imgx, imgy = self.image.width(), self.image.height()
            xfrac, yfrac = imgx/x, imgy/y
            w = x * 0.8 if xfrac == yfrac else imgx * 0.8 * 1/yfrac
            self.scaled_image_width = int(w)
            self.scale_factor = imgx/w
        self.image = self.image.scaledToWidth(self.scaled_image_width)
        self.parent.setGeometry(30, 30, self.image.width(), self.image.height())
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        self.resize(self.image.width(), self.image.height())
        qp.drawPixmap(self.rect(), self.image)
        br = QBrush(QColor(100, 10, 10, 10))  
        pen = QPen(QColor(0, 0, 0, 255), 1.5)
        qp.setBrush(br)
        qp.setPen(pen)

    def mousePressEvent(self, event):
        x, y = event.pos().x(), event.pos().y()
        x, y = self.convert_coords(x, y)
        self.x, self.y = x, y
        if self.mode==0:
            self.select_barcol()
        if self.mode==1:
            self.apply_barbox(x, y)
        if self.mode==2:
            self.apply_crop(x, y)
        if self.mode==3:
            self.add_phase()
        if self.mode==5:
            self.set_voxsize(x, y)

    def convert_coords(self, x, y):
        x0, y0 = self.img.shape[:2]
        w = self.frameGeometry().width()
        h = self.frameGeometry().height()
        x = int(x*x0/h)
        y = int(y*y0/w)
        return x, y
    


    def apply_crop(self, x, y):
            if len(self.crop)<2:
                self.crop.append((x, y))
                if len(self.crop)==2:
                    x0, y0 = self.crop[0]
                    self.img = self.img[y0:y, x0:x]
                    self.load_temp()
            else:
                self.crop = []
                self.img = plt.imread(self.img_path)
                self.setPixmap(self.img_path)

    def set_voxsize(self, x, y):
        if len(self.voxsize)<2:
            self.voxsize.append(x)
        if len(self.voxsize)==2:
            self.get_sb()
    def get_sb(self):
        num,ok = QInputDialog.getInt(self,"enter the scalebar value in microns","enter the scalebar value in microns")
        if ok:
            self.sb = (int(num))
            
    def apply_barbox(self, x, y):
        if len(self.bar_box)<2:
            self.bar_box.append((x, y))
            if len(self.bar_box)==2:
                x0, y0 = self.bar_box[0]
                self.img[y0:y, x0:x] -= 0.1
                self.img[self.img< 0] = 0
                self.load_temp()
        else:
            self.bar_box = []
            self.reload_img()

    def select_barcol(self):
        self.reload_img()
        self.bar_col = self.img[self.y, self.x]
        mask = np.where((abs(self.img-self.bar_col)<self.boundary).all(axis=2))
        self.img[mask] = [1,0,0,1]
        self.load_temp()
        
    def add_phase(self):
        ph = self.img[self.y, self.x, 0]
        self.phases.append(ph)
        self.phases = sorted(self.phases)
        self.show_phases()

    def rotateimg(self):
        self.reload_img()
        self.img = ndimage.rotate(self.img, self.rot, reshape=False)
        self.img[self.img > 1] = 1
        self.img[self.img < 0] = 0
        self.load_temp()

    def show_phases(self):
        self.reload_img()
        if len(self.phases)==1:
            self.img[...,0][self.img[...,0] < self.phases[0]] +=0.1 
        else:
            self.phases = sorted(self.phases)
            boundaries = [0]
            for ph1, ph2 in zip(self.phases[:-1], self.phases[1:]):
                boundaries.append(ph1 + (ph2 - ph1)/2)
            boundaries.append(1)
            for i, (b_low, b_high) in enumerate(zip(boundaries[:-1],boundaries[1:])):
                self.img[..., i][(self.img[...,i] >= b_low) & (self.img[...,i] <= b_high)] += 1 if self.opaque else 0
            
        self.img[self.img>1] = 1
        self.img[self.img<0] = 0
        self.load_temp()

    def reload_img(self):
        self.img = plt.imread(self.img_path)
        self.setPixmap(self.img_path)

    def update_current_img(self):
        self.current_img += 1
        while self.current_img in self.good:
            self.current_img +=1
        print(self.current_img)

    def load_temp(self):
        plt.imsave('data/temp.png', self.img)
        self.setPixmap('data/temp.png')


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
                
            self.mode += 1
            
            if self.mode==6:
                info = {}
                pth = self.img_path
                info['data_path'] = pth
                info['crop'] = self.crop
                info['barbox'] = self.bar_box
                info['voxsize'] = self.sb / (self.voxsize[1] - self.voxsize[0])
                info['barcol'] = (int(self.bar_col[0]*255), int(self.boundary*1000)) if self.bar_col[0] != None else None
                info['phases'] = [int(ph*255) for ph in self.phases] 
                info['rot'] = self.rot
                info['data_type'] = 'grayscale' if len(self.phases) == 0 else 'twophase'
                
                key = len(self.data_map.keys()) + 1
                self.data_map[key] = info
                with open(f'data/anns.json', 'w') as f:
                    json.dump(self.data_map, f)
                self.clear()
                self.update_current_img()
                self.img_name = self.img_list[self.current_img]
                self.img_path = f'{self.data_path}/{self.img_name}'
            print(f'{self.current_img}/{len(self.img_list)} complete')
            self.reload_img()

        elif event.key() == Qt.Key_C:
            self.clear()
            self.reload_img()


        elif event.key() == Qt.Key_W:
            try:
                self.data_map.pop(str(self.current_img))
            except:
                pass
            with open(f'data/anns.json', 'w') as f:
                json.dump(self.data_map, f)
            self.clear()
            self.update_current_img()
            self.img_name = self.img_list[self.current_img]
            self.img_path = f'{self.data_path}/{self.img_name}'
            self.reload_img()
            
        elif event.key() == Qt.Key_Q:
            self.mode = 'skip'
        elif event.key() == Qt.Key_A:
            if self.mode==0:
                self.boundary +=0.01
                self.select_barcol()

            if self.mode==3:
                if self.phases[self.selected_phase] < 1:
                    self.phases[self.selected_phase] +=0.02
                    self.show_phases()

            if self.mode==4:
                self.rot+=2
                self.rotateimg()
                

        elif event.key() == Qt.Key_S:
            if self.mode==0:
                self.boundary -=0.01
                self.select_barcol()

            if self.mode==3:
                if self.phases[self.selected_phase] > 0:
                    self.phases[self.selected_phase] -=0.02
                    self.show_phases()

            if self.mode==4:
                self.rot-=2
                self.rotateimg()

        elif event.text().isnumeric():
            self.selected_phase = int(event.text()) - 1
            
        elif event.key() == Qt.Key_X:
            self.opaque = 0 if self.opaque else 1
            self.show_phases()

    def clear(self):
        self.crop = []
        self.voxsize = []
        self.bar_box = []
        self.bar_col = [None]
        self.phases = []
        self.rot = 0
        self.mode = 0
        self.stage.setText(self.modes[self.mode])

        self.reload_img()


def preprocess_gui():

    app = QApplication(sys.argv)
    qss="src/style.qss"
    with open(qss,"r") as fh:
        app.setStyleSheet(fh.read())
    window = MainWindow(app)

    sys.exit(app.exec_())
