import sys
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QMessageBox
from PyQt5.QtWidgets import QDesktopWidget
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
            self.resize(500, 150)
            self.move(100, 100)
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
            self.setToolTip("this is widget")

            btn = QPushButton("Button", self)  # self类似于C++ this指针
            btn.setToolTip("This is a button")
            btn.resize(btn.sizeHint())
            btn.move(0, 0)

            btn2 = QPushButton("Button2", self)  # self类似于C++ this指针
            btn2.setToolTip("This is second button")
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
    


if __name__ == '__main__':
    test3()
    