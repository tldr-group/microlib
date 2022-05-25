from logging.handlers import RotatingFileHandler
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QAction, QComboBox, QLabel, QLineEdit
from PyQt5.QtGui import QColor, QBrush, QPainter, QPixmap, QPolygonF, QPen
from PyQt5.QtCore import QPoint, QRect, QPointF, QThread, QTimeLine, Qt
import matplotlib.pyplot as plt
from sympy import re
import numpy as np
import os
import json
from scipy import ndimage

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
        self.data_path = 'data/micrographs_png'
        self.sb = QLineEdit(self)
        selector = parent.addToolBar("Image Type")
        selector.addWidget(self.sb)
        
        with open(f'data/data_map.json', 'r') as f:
                    self.data_map = json.load(f)
                    print('loading')
        self.fresh_data_map = {}
        self.crop = []
        self.current_img = -1
        self.reset()
        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Name:')
        
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
        self.apply_crop(x, y)
        

    def convert_coords(self, x, y):
        x0, y0 = self.img.shape[:2]
        w = self.frameGeometry().width()
        h = self.frameGeometry().height()
        x = int(x*x0/h)
        y = int(y*y0/w)
        return x, y
    


    def apply_crop(self, x, y):
            if len(self.crop)<2:
                self.crop.append(x)
                
            else:
                self.crop = []
                

    
    def reload_img(self):
        self.img = plt.imread(self.img_path)
        self.setPixmap(self.img_path)


    def load_temp(self):
        plt.imsave('data/temp.png', self.img)
        self.setPixmap('data/temp.png')

    def mouseDoubleClickEvent(self,event):
        return

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            
            n = int(self.sb.text())
            info = {}
            pth = self.img_path
            pixels = self.crop[1] - self.crop[0]
            voxsize = n/pixels
            print(self.current_img, f'{pixels} pixels is {n} micros giving a voxel size of {voxsize}') 
            self.sample['vox_size'] = voxsize
            self.fresh_data_map[str(self.current_img)] = self.sample
            with open(f'data/data_map_vs.json', 'w') as f:
                json.dump(self.fresh_data_map, f)
            self.reset()


    def reset(self):
        try:
            self.current_img+=1
            self.sb.setText('')
            self.sample = self.data_map[str(self.current_img)]
            self.img_path = self.sample['data_path']
            self.img = plt.imread(self.img_path)
            self.setPixmap(self.img_path, loading=True)
            self.crop = []
        except:
            print('failed')
            if self.current_img > 1000:
                return
            else:
                self.reset()
                




        
    

def main():

    app = QApplication(sys.argv)
    qss="style.qss"
    with open(qss,"r") as fh:
        app.setStyleSheet(fh.read())
    window = MainWindow(app)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()