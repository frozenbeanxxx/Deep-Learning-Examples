import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QAction, qApp
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QIcon
 
def t1():
    class Example(QMainWindow):
        
        def __init__(self):
            super().__init__()
            self.initUI()
            
        def initUI(self): 
            self.statusBar().showMessage('Ready')
            self.setGeometry(300, 300, 250, 150)
            self.setWindowTitle('Statusbar')    
            self.show()
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

def t2():
    class Example(QMainWindow):
        def __init__(self):
            super().__init__()
            self.initUI()
        def initUI(self):
            exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
            exitAction.setShortcut('Ctrl+Q')
            exitAction.setStatusTip('Exit application')
            exitAction.triggered.connect(qApp.quit)
    
            self.statusBar()
    
            #创建一个菜单栏
            menubar = self.menuBar()
            #添加菜单
            fileMenu = menubar.addMenu('&File')
            #添加事件
            fileMenu.addAction(exitAction)
            
            self.setGeometry(300, 300, 300, 200)
            self.setWindowTitle('Menubar')    
            self.show()

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

def t3():
    class Example(QMainWindow):
        def __init__(self):
            super().__init__()
            self.initUI()
        def initUI(self):
            exitAction = QAction(QIcon('exit24.png'), 'Exit', self)
            exitAction.setShortcut('Ctrl+Q')
            exitAction.triggered.connect(qApp.quit)
            
            self.toolbar = self.addToolBar('Exit')
            self.toolbar.addAction(exitAction)
            
            self.setGeometry(300, 300, 300, 200)
            self.setWindowTitle('Toolbar')    
            self.show()

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

def t4():
    class Example(QMainWindow):
        def __init__(self):
            super().__init__()
            self.initUI()
        def initUI(self):
            textEdit = QTextEdit()
            self.setCentralWidget(textEdit)
    
            exitAction = QAction(QIcon('exit24.png'), 'Exit', self)
            exitAction.setShortcut('Ctrl+Q')
            exitAction.setStatusTip('Exit application')
            exitAction.triggered.connect(self.close)
    
            self.statusBar()
    
            menubar = self.menuBar()
            fileMenu = menubar.addMenu('&File')
            fileMenu.addAction(exitAction)
    
            toolbar = self.addToolBar('Exit')
            toolbar.addAction(exitAction)
            
            self.setGeometry(300, 300, 350, 250)
            self.setWindowTitle('Main window')    
            self.show()
        
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

if __name__ == '__main__':
    t4()
    