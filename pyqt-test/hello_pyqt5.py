import sys
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QMessageBox, QLabel
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QLineEdit, QTextEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QCoreApplication


def test1():
    app = QApplication(sys.argv)  # 创建一个应用

    w = QWidget()  # 在这个应用条件下创建一个窗口
    w.resize(500, 150)  # 重新设置大小
    w.move(200, 400)  # 移动下距离
    w.setWindowTitle("Hello world!")  # 设置标题
    w.setWindowIcon(QIcon('./Title.ico'))
    w.show()  # 显示这个窗口
    
    sys.exit(app.exec_())  # 将app的退出信号传给程序进程，让程序进程退出

def test2():
    class Example(QWidget):
        def __init__(self):
            super().__init__()
            self.setUI()

        def setUI(self):
            #self.resize(500, 150)
            #self.move(100, 100)
            self.setGeometry(300,400,500,600)
            self.setWindowIcon(QIcon('./Title.ico'))
            self.setWindowTitle("Hello world")
            self.show()


    app = QApplication(sys.argv)

    ex = Example()

    sys.exit(app.exec_())

def test3():
    class Example(QWidget):
        def __init__(self):
            super().__init__()
            self.setUI()

        def setUI(self):
            QToolTip.setFont(QFont('SansSerif', 10))
            self.resize(500, 150)
            self.move(100, 100)
            self.setWindowIcon(QIcon('./Title.ico'))
            self.setWindowTitle("Hello world")

            #self.setToolTip("<b>this is widget</b>")
            self.setToolTip("this is <b>QWidget</b> widget")

            btn = QPushButton("Button", self)  # self类似于C++ this指针
            btn.setToolTip("This is a button")
            btn.resize(btn.sizeHint())
            btn.move(0, 0)

            btn2 = QPushButton("Button2", self)  # self类似于C++ this指针
            btn2.setToolTip("This is <b>second</b> button")
            btn2.resize(btn2.sizeHint())
            btn2.move(100, 0)

            btn = QPushButton("quit Button", self)  # self类似于C++ this指针
            btn.setToolTip("This is a button will quit itself")
            btn.clicked.connect(QCoreApplication.instance().quit)
            btn.resize(btn.sizeHint())
            btn.move(100, 50)

            self.center()
            self.show()
        
        def closeEvent(self, event):
            reply = QMessageBox.question(self, 'Message',
                                        "Are you sure to quit?", QMessageBox.Yes |
                                        QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

        def center(self):
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())


    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_()) 
    
def t4():
    # 绝对像素坐标布局
    # 标签使用
    # 字符串属性
    class Example(QWidget):
        def __init__(self):
            super().__init__()
            self.initUI()

        def initUI(self):
            lbl1 = QLabel('Zetcode', self)
            lbl1.move(15, 10)
    
            lbl2 = QLabel('<p><font color=red><b>tutorials</b></font></p>', self)
            lbl2.move(35, 40)
            
            lbl3 = QLabel('for programmers', self)
            lbl3.move(55, 70)        
            
            self.setGeometry(300, 300, 250, 150)
            self.setWindowTitle('Absolute')    
            self.show()
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

def t5():
    # 相对位置布局
    # 横向布局，纵向布局，伸展因子
    class Example(QWidget):
        def __init__(self):
            super().__init__()
            self.initUI()

        def initUI(self):
            okButton = QPushButton("OK")
            cancelButton = QPushButton("Cancel")
    
            hbox = QHBoxLayout()
            hbox.addStretch(1)
            hbox.addWidget(okButton)
            hbox.addWidget(cancelButton)
    
            vbox = QVBoxLayout()
            vbox.addStretch(1)
            vbox.addLayout(hbox)
            
            self.setLayout(vbox)    
            
            self.setGeometry(300, 300, 300, 150)
            self.setWindowTitle('Buttons')    
            self.show()

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

def t6():
    # 表格布局，适合计算器界面
    class Example(QWidget):
        def __init__(self):
            super().__init__()
            self.initUI()
        def initUI(self):
            grid = QGridLayout()
            self.setLayout(grid)
            names = ['Cls', 'Bck', '', 'Close',
                    '7', '8', '9', '/',
                    '4', '5', '6', '*',
                    '1', '2', '3', '-',
                    '0', '.', '=', '+']
            positions = [(i,j) for i in range(5) for j in range(4)]
            for position, name in zip(positions, names):
                if name == '':
                    continue
                button = QPushButton(name)
                grid.addWidget(button, *position)
            self.move(300, 150)
            self.setWindowTitle('Calculator')
            self.show()
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

def t7():
    class Example(QWidget):
        def __init__(self):
            super().__init__()
            self.initUI()

        def initUI(self):
            
            title = QLabel('Title')
            author = QLabel('Author')
            review = QLabel('Review')
    
            titleEdit = QLineEdit()
            authorEdit = QLineEdit()
            reviewEdit = QTextEdit()
    
            grid = QGridLayout()
            grid.setSpacing(10)
    
            grid.addWidget(title, 1, 0)
            grid.addWidget(titleEdit, 1, 1)
    
            grid.addWidget(author, 2, 0)
            grid.addWidget(authorEdit, 2, 1)
    
            grid.addWidget(review, 3, 0)
            grid.addWidget(reviewEdit, 3, 1, 5, 1)

            button1 = QPushButton("button1", self)
            grid.addWidget(button1, 10, 0)
            
            self.setLayout(grid) 
            
            self.setGeometry(300, 300, 350, 300)
            self.setWindowTitle('Review')    
            self.show()

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

if __name__ == '__main__':
    t7()
    