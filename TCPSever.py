from time import ctime, sleep
import socket
import sys
from UILib.MainWindow import MainWindow
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import qdarkstyle
from PyQt5.QtWidgets import QApplication
from UILib.Database import KEYS
from PyQt5.QtGui import QImage, QPixmap


class mythread(QThread):  # 步骤1.创建一个线程实例
    mysignal = pyqtSignal(str)  # 创建一个自定义信号，元组参数

    def __init__(self, ipText, portValue):
        super(mythread, self).__init__()
        self.ipText = ipText
        self.portValue = portValue

    def run(self):
        # 套接字类型AF_INET, socket.SOCK_STREAM   tcp协议，基于流式的协议
        self.ser = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 对socket的配置重用ip和端口号
        self.ser.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定端口号
        self.ser.bind((self.ipText, self.portValue))  # 写哪个ip就要运行在哪台机器上
        # 设置半连接池
        self.ser.listen(3)  # 最多可以连接多少个客户端
        print('【Sever启动成功】{}\n目标IP：{}\n端口号：{}'.format(
            ctime(), self.ipText, self.portValue))
        while 1:
            # 阻塞等待，创建连接
            print("等待连接中..")
            con, address = self.ser.accept()  # 在这个位置进行等待，监听端口号
            print("等待消息中...")
            while 1:
                try:
                    # 接受套接字的大小，怎么发就怎么收
                    msg = con.recv(1024)
                    meg = msg.decode('utf-8')
                    if meg == 'quit':
                        # 断开连接
                        con.close()
                    self.mysignal.emit(meg)  # 发射自定义信号
                except Exception as e:
                    print(e)
                    break


class MainSeverUI(MainWindow):

    def __init__(self):
        """
        初始化界面 ，连接槽函数，以及设置校验器
        """
        super(MainSeverUI, self).__init__()
        self.values_max_num = 50
        self.face_pt = None
        self.sever_num = 0
        self.my_thread = [None, None, None]

    def update_values_1(self, values):
        try:
            if len(values):
                values = values.split(',')
                self.graph_values[0].append(eval(values[0]))
                self.eye_values[0].append(eval(values[1]))
                if values[2] == '0':
                    self.face_pt = None
                else:
                    self.face_pt = values[2]
                    self.updateSeverLog(self.face_pt)
            if len(self.graph_values[0]) > self.values_max_num:
                self.graph_values[0].pop(0)
            if len(self.eye_values[0]) > self.values_max_num:
                self.eye_values[0].pop(0)
        except Exception as e:
            print(e)
            self.my_thread[0].quit()

    def update_values_2(self, values):
        try:
            if len(values):
                values = values.split(',')
                self.graph_values[1].append(eval(values[0]))
                self.eye_values[1].append(eval(values[1]))
                if values[2] == '0':
                    self.face_pt = None
                else:
                    self.face_pt = values[2]
                    self.updateSeverLog(self.face_pt)
            if len(self.graph_values[1]) > self.values_max_num:
                self.graph_values[1].pop(0)
            if len(self.eye_values[1]) > self.values_max_num:
                self.eye_values[1].pop(0)
        except Exception as e:
            print(e)
            self.my_thread[1].quit()

    def update_values_3(self, values):
        try:
            if len(values):
                values = values.split(',')
                self.graph_values[2].append(eval(values[0]))
                self.eye_values[2].append(eval(values[1]))
                if values[2] == '0':
                    self.face_pt = None
                else:
                    self.face_pt = values[2]
                    self.updateSeverLog(self.face_pt)
            if len(self.graph_values[2]) > self.values_max_num:
                self.graph_values[2].pop(0)
            if len(self.eye_values[2]) > self.values_max_num:
                self.eye_values[2].pop(0)
        except Exception as e:
            print(e)
            self.my_thread[2].quit()

    def updateSeverLog(self, face_pt):
        print('【检测成功】')
        value = {KEYS.CARID: '梁瑛平',
                 KEYS.CARIMAGE: QPixmap(face_pt),
                 KEYS.CARCOLOR: '男',
                 KEYS.LICENSEIMAGE: None,
                 KEYS.LICENSENUMBER: '1120182525',
                 KEYS.LOCATION: str(self.camera_group.currentText()),
                 KEYS.RULENAME: '待输入',
                 KEYS.RULEID: '待输入'}
        self.updateLog(value)

    def start_tcp_server(self):
        # 实例化一个socket
        ipText = self.myip
        if self.sever_num == 0:
            if self.address1.text() == '待输入':
                print('【错误】请输入正确的IP地址和端口号')
                return
            else:
                portValue = int(self.address1.text())
        elif self.sever_num == 1:
            if self.address2.text() == '待输入':
                print('【错误】请输入正确的IP地址和端口号')
                return
            else:
                portValue = int(self.address2.text())
        else:
            if self.address2.text() == '待输入':
                print('【错误】请输入正确的IP地址和端口号')
                return
            else:
                portValue = int(self.address2.text())

        self.my_thread[self.sever_num] = mythread(ipText, portValue)  # 主线程连接子线
        if self.sever_num == 0:
            self.my_thread[self.sever_num].mysignal.connect(
                self.update_values_1)  # 自定义信号连接
        elif self.sever_num == 1:
            self.my_thread[self.sever_num].mysignal.connect(
                self.update_values_2)  # 自定义信号连接
        else:
            self.my_thread[self.sever_num].mysignal.connect(
                self.update_values_3)  # 自定义信号连接
        print('【线程启动】{}\n目标IP：{}\n端口号：{}'.format(
            ctime(), ipText, portValue))
        self.my_thread[self.sever_num].start()  # 子线程开始执行run函数
        self.sever_num += 1


def main():
    '''
    启动PyQt5程序，打开GUI界面。
    '''
    app = QApplication(sys.argv)
    main_window = MainSeverUI()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window.generate_image()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    main()