from time import ctime, sleep
import socket
import sys
import cv2

from UILib.SonWindow import MainWindow
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QSizePolicy
from processor.MainProcessor import MainProcessor


class MainSocketUI(MainWindow):

    def __init__(self):
        """
        初始化界面 ，连接槽函数，以及设置校验器
        """
        super(MainSocketUI, self).__init__()
        self.label.setText('智能医疗监控云平台——终端')
        self.resize(450, 600)
        self.p = None

        if self.myip is None:
            self.target_ip.setText('异常，无法获取本机IP')
        else:
            self.target_ip.setText(str(self.myip))
        self.vs = cv2.VideoCapture(0)
        self.video = None
        ret, _ = self.vs.read()
        if not ret:
            print('【进程错误】摄像头被占用，启用默认视频')
            self.vs = cv2.VideoCapture('data/peng.mp4')
            self.video = 'data/peng.mp4'
        self.address1.setText('待输入')
        self.close_extra_btn()

    def close_extra_btn(self):

        pass

    def start_tcp_server(self):
        # 实例化一个socket
        ipText = self.myip
        if self.address1.text() == '待输入':
            print('【错误】请输入正确的IP地址和端口号')
            return
        else:
            portValue = int(self.address1.text())

        self.p = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.p.connect((ipText, portValue))
        print('【连接成功】{}\n目标IP：{}\n端口号：{}'.format(
            ctime(), ipText, portValue))
        self.processor = MainProcessor(None, False, p=self.p)

def main():
    '''
    启动PyQt5程序，打开GUI界面。
    '''
    app = QApplication(sys.argv)
    main_window = MainSocketUI()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    import qdarkstyle
    from PyQt5.QtWidgets import QApplication

    main()
