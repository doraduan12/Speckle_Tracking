# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(675, 724)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_browse = QtWidgets.QPushButton(self.centralwidget)
        self.btn_browse.setGeometry(QtCore.QRect(10, 10, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        self.btn_browse.setFont(font)
        self.btn_browse.setObjectName("btn_browse")
        self.textBrowser_browse = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_browse.setGeometry(QtCore.QRect(100, 10, 551, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        self.textBrowser_browse.setFont(font)
        self.textBrowser_browse.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.textBrowser_browse.setObjectName("textBrowser_browse")
        self.groupBox_filedetail = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_filedetail.setGeometry(QtCore.QRect(460, 50, 201, 251))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        self.groupBox_filedetail.setFont(font)
        self.groupBox_filedetail.setObjectName("groupBox_filedetail")
        self.label_image_size = QtWidgets.QLabel(self.groupBox_filedetail)
        self.label_image_size.setGeometry(QtCore.QRect(10, 30, 81, 21))
        self.label_image_size.setTextFormat(QtCore.Qt.AutoText)
        self.label_image_size.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_image_size.setObjectName("label_image_size")
        self.textBrowser_image_size = QtWidgets.QTextBrowser(self.groupBox_filedetail)
        self.textBrowser_image_size.setGeometry(QtCore.QRect(90, 30, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textBrowser_image_size.setFont(font)
        self.textBrowser_image_size.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.textBrowser_image_size.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.textBrowser_image_size.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_image_size.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_image_size.setObjectName("textBrowser_image_size")
        self.textBrowser_datetime = QtWidgets.QTextBrowser(self.groupBox_filedetail)
        self.textBrowser_datetime.setGeometry(QtCore.QRect(90, 70, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textBrowser_datetime.setFont(font)
        self.textBrowser_datetime.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.textBrowser_datetime.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.textBrowser_datetime.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_datetime.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_datetime.setObjectName("textBrowser_datetime")
        self.label_datetime = QtWidgets.QLabel(self.groupBox_filedetail)
        self.label_datetime.setGeometry(QtCore.QRect(10, 80, 81, 21))
        self.label_datetime.setTextFormat(QtCore.Qt.AutoText)
        self.label_datetime.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_datetime.setObjectName("label_datetime")
        self.label_frame = QtWidgets.QLabel(self.groupBox_filedetail)
        self.label_frame.setGeometry(QtCore.QRect(10, 130, 81, 21))
        self.label_frame.setTextFormat(QtCore.Qt.AutoText)
        self.label_frame.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_frame.setObjectName("label_frame")
        self.textBrowser_frame = QtWidgets.QTextBrowser(self.groupBox_filedetail)
        self.textBrowser_frame.setGeometry(QtCore.QRect(90, 130, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textBrowser_frame.setFont(font)
        self.textBrowser_frame.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.textBrowser_frame.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.textBrowser_frame.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_frame.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_frame.setReadOnly(True)
        self.textBrowser_frame.setObjectName("textBrowser_frame")
        self.textBrowser_delta_x = QtWidgets.QTextBrowser(self.groupBox_filedetail)
        self.textBrowser_delta_x.setGeometry(QtCore.QRect(90, 170, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textBrowser_delta_x.setFont(font)
        self.textBrowser_delta_x.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.textBrowser_delta_x.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.textBrowser_delta_x.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_delta_x.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_delta_x.setReadOnly(False)
        self.textBrowser_delta_x.setObjectName("textBrowser_delta_x")
        self.label_delta_x = QtWidgets.QLabel(self.groupBox_filedetail)
        self.label_delta_x.setGeometry(QtCore.QRect(10, 170, 81, 21))
        self.label_delta_x.setTextFormat(QtCore.Qt.AutoText)
        self.label_delta_x.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_delta_x.setObjectName("label_delta_x")
        self.textBrowser_delta_y = QtWidgets.QTextBrowser(self.groupBox_filedetail)
        self.textBrowser_delta_y.setGeometry(QtCore.QRect(90, 210, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textBrowser_delta_y.setFont(font)
        self.textBrowser_delta_y.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.textBrowser_delta_y.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.textBrowser_delta_y.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_delta_y.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_delta_y.setReadOnly(False)
        self.textBrowser_delta_y.setObjectName("textBrowser_delta_y")
        self.label_delta_y = QtWidgets.QLabel(self.groupBox_filedetail)
        self.label_delta_y.setGeometry(QtCore.QRect(10, 210, 81, 21))
        self.label_delta_y.setTextFormat(QtCore.Qt.AutoText)
        self.label_delta_y.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_delta_y.setObjectName("label_delta_y")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 430, 431, 231))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setGeometry(QtCore.QRect(450, 630, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        self.btn_save.setFont(font)
        self.btn_save.setObjectName("btn_save")
        self.textBrowser_save = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_save.setGeometry(QtCore.QRect(450, 590, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        self.textBrowser_save.setFont(font)
        self.textBrowser_save.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.textBrowser_save.setObjectName("textBrowser_save")
        self.groupBox_preview = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_preview.setGeometry(QtCore.QRect(10, 50, 431, 371))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        self.groupBox_preview.setFont(font)
        self.groupBox_preview.setObjectName("groupBox_preview")
        self.label_preview = QtWidgets.QLabel(self.groupBox_preview)
        self.label_preview.setGeometry(QtCore.QRect(10, 30, 411, 291))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        self.label_preview.setFont(font)
        self.label_preview.setStyleSheet("background-color: rgba(220, 220, 220, 200);")
        self.label_preview.setText("")
        self.label_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.label_preview.setObjectName("label_preview")
        self.btn_run = QtWidgets.QPushButton(self.groupBox_preview)
        self.btn_run.setGeometry(QtCore.QRect(320, 330, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(10)
        self.btn_run.setFont(font)
        self.btn_run.setObjectName("btn_run")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 675, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_browse.setText(_translate("MainWindow", "Browse"))
        self.groupBox_filedetail.setTitle(_translate("MainWindow", "File Detail"))
        self.label_image_size.setText(_translate("MainWindow", "Image size"))
        self.label_datetime.setText(_translate("MainWindow", "Datetime"))
        self.label_frame.setText(_translate("MainWindow", "Frame"))
        self.label_delta_x.setText(_translate("MainWindow", "Delta X"))
        self.label_delta_y.setText(_translate("MainWindow", "Delta Y"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.btn_save.setText(_translate("MainWindow", "Save Result"))
        self.groupBox_preview.setTitle(_translate("MainWindow", "Preview"))
        self.btn_run.setText(_translate("MainWindow", "Run"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())