from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from python_speech_features import mfcc
from python_speech_features import logfbank
import librosa
from pysndfx import AudioEffectsChain
import numpy as np
import math
import scipy.io.wavfile as wav
import pyaudio
import datetime
import wave
import os

from threading import Thread
import time as t
import sqlite3
import serious as conv
obj2 = conv.conversation()
from python_speech_features import fbank, delta, mfcc
import numpy as np
import librosa
import sounddevice as sd
import FunctionsToIntegrate as fnt
obj = fnt.model_holder()
conn = sqlite3.connect('users.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS USERS
         (ID INT PRIMARY KEY NOT NULL,
         NAME TEXT NOT NULL,
         AGE INT NOT NULL,
         SEX TEXT,
         VOICE TEXT,
         PROFILE TEXT,
         REP TEXT);''')
cursor = conn.cursor()

au_format = pyaudio.paInt16
no_channels = 2
chunk = 1024
sampling_rate = 48000
sample_duration = 3

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(550, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(550, 500))
        MainWindow.setMaximumSize(QtCore.QSize(550, 500))
        MainWindow.setAutoFillBackground(False)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 551, 501))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 270, 521, 52))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMaximumSize(QtCore.QSize(50, 50))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("record.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.timer = QtWidgets.QLCDNumber(self.horizontalLayoutWidget_2)
        self.timer.setMaximumSize(QtCore.QSize(70, 50))
        self.timer.setObjectName("timer")
        self.horizontalLayout_2.addWidget(self.timer)
        self.recordButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.recordButton.sizePolicy().hasHeightForWidth())
        self.recordButton.setSizePolicy(sizePolicy)
        self.recordButton.setObjectName("recordButton")
        self.horizontalLayout_2.addWidget(self.recordButton)
        self.stopButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stopButton.sizePolicy().hasHeightForWidth())
        self.stopButton.setSizePolicy(sizePolicy)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_2.addWidget(self.stopButton)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 320, 521, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.chooseButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chooseButton.sizePolicy().hasHeightForWidth())
        self.chooseButton.setSizePolicy(sizePolicy)
        self.chooseButton.setObjectName("chooseButton")
        self.horizontalLayout.addWidget(self.chooseButton)
        self.file_path_1 = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_path_1.sizePolicy().hasHeightForWidth())
        self.file_path_1.setSizePolicy(sizePolicy)
        self.file_path_1.setReadOnly(True)
        self.file_path_1.setObjectName("file_path_1")
        self.horizontalLayout.addWidget(self.file_path_1)
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(10, 240, 531, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 370, 521, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.saveButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())
        self.saveButton.setSizePolicy(sizePolicy)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout_3.addWidget(self.saveButton)
        self.trainButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trainButton.sizePolicy().hasHeightForWidth())
        self.trainButton.setSizePolicy(sizePolicy)
        self.trainButton.setObjectName("trainButton")
        self.horizontalLayout_3.addWidget(self.trainButton)
        self.quitButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.quitButton.sizePolicy().hasHeightForWidth())
        self.quitButton.setSizePolicy(sizePolicy)
        self.quitButton.setObjectName("quitButton")
        self.horizontalLayout_3.addWidget(self.quitButton)
        self.uploadButton = QtWidgets.QPushButton(self.tab)
        self.uploadButton.setGeometry(QtCore.QRect(20, 190, 121, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.uploadButton.sizePolicy().hasHeightForWidth())
        self.uploadButton.setSizePolicy(sizePolicy)
        self.uploadButton.setObjectName("uploadButton")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(10, 0, 531, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(20, 40, 121, 131))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("profile.png"))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(160, 40, 371, 51))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_6 = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.label_6.setScaledContents(True)
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.switch_user = QtWidgets.QComboBox(self.horizontalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.switch_user.sizePolicy().hasHeightForWidth())
        self.switch_user.setSizePolicy(sizePolicy)
        self.switch_user.setObjectName("switch_user")
        self.switch_user.clear()
        cursor = conn.execute("SELECT NAME from USERS")
        alist = ["[New User]"]
        for row in cursor:
            alist.extend(row)
        self.switch_user.addItems(alist)
        self.switch_user.currentIndexChanged.connect(self.selectionchange)
        self.horizontalLayout_4.addWidget(self.switch_user)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(160, 90, 371, 51))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_7 = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        self.label_7.setScaledContents(True)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.name_edit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.name_edit.sizePolicy().hasHeightForWidth())
        self.name_edit.setSizePolicy(sizePolicy)
        self.name_edit.setReadOnly(False)
        self.name_edit.setObjectName("name_edit")
        self.horizontalLayout_5.addWidget(self.name_edit)
        self.horizontalLayoutWidget_9 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_9.setGeometry(QtCore.QRect(160, 140, 371, 51))
        self.horizontalLayoutWidget_9.setObjectName("horizontalLayoutWidget_9")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_9)
        self.horizontalLayout_9.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_9.setSpacing(6)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(6)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_9 = QtWidgets.QLabel(self.horizontalLayoutWidget_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setScaledContents(True)
        self.label_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_9.setWordWrap(True)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_7.addWidget(self.label_9)
        self.sex_select = QtWidgets.QComboBox(self.horizontalLayoutWidget_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sex_select.sizePolicy().hasHeightForWidth())
        self.sex_select.setSizePolicy(sizePolicy)
        self.sex_select.setObjectName("sex_select")
        self.sex_select.addItem("")
        self.sex_select.addItem("")
        self.horizontalLayout_7.addWidget(self.sex_select)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_8 = QtWidgets.QLabel(self.horizontalLayoutWidget_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setScaledContents(True)
        self.label_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_6.addWidget(self.label_8)
        self.age_select = QtWidgets.QSpinBox(self.horizontalLayoutWidget_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.age_select.sizePolicy().hasHeightForWidth())
        self.age_select.setSizePolicy(sizePolicy)
        self.age_select.setMinimum(1)
        self.age_select.setObjectName("age_select")
        self.horizontalLayout_6.addWidget(self.age_select)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_6)
        self.horizontalLayoutWidget_8 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_8.setGeometry(QtCore.QRect(160, 190, 371, 51))
        self.horizontalLayoutWidget_8.setObjectName("horizontalLayoutWidget_8")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_8)
        self.horizontalLayout_8.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_8.setSpacing(6)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.clearButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clearButton.sizePolicy().hasHeightForWidth())
        self.clearButton.setSizePolicy(sizePolicy)
        self.clearButton.setObjectName("clearButton")
        self.horizontalLayout_8.addWidget(self.clearButton)
        self.updateButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.updateButton.sizePolicy().hasHeightForWidth())
        self.updateButton.setSizePolicy(sizePolicy)
        self.updateButton.setObjectName("updateButton")
        self.horizontalLayout_8.addWidget(self.updateButton)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setGeometry(QtCore.QRect(10, 0, 531, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.horizontalLayoutWidget_11 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_11.setGeometry(QtCore.QRect(10, 30, 521, 51))
        self.horizontalLayoutWidget_11.setObjectName("horizontalLayoutWidget_11")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_11)
        self.horizontalLayout_12.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_12.setSpacing(6)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.newTaskButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.newTaskButton.sizePolicy().hasHeightForWidth())
        self.newTaskButton.setSizePolicy(sizePolicy)
        self.newTaskButton.setMaximumSize(QtCore.QSize(16777215, 100))
        self.newTaskButton.setObjectName("newTaskButton")
        self.horizontalLayout_12.addWidget(self.newTaskButton)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setSpacing(6)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.checkBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget_11)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_11.addWidget(self.checkBox)
        self.identity_field = QtWidgets.QLineEdit(self.horizontalLayoutWidget_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.identity_field.sizePolicy().hasHeightForWidth())
        self.identity_field.setSizePolicy(sizePolicy)
        self.identity_field.setReadOnly(True)
        self.identity_field.setObjectName("identity_field")
        self.horizontalLayout_11.addWidget(self.identity_field)
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12.addLayout(self.verticalLayout)
        self.horizontalLayoutWidget_12 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_12.setGeometry(QtCore.QRect(10, 100, 521, 51))
        self.horizontalLayoutWidget_12.setObjectName("horizontalLayoutWidget_12")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_12)
        self.horizontalLayout_13.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_13.setSpacing(6)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_13 = QtWidgets.QLabel(self.horizontalLayoutWidget_12)
        self.label_13.setMaximumSize(QtCore.QSize(50, 50))
        self.label_13.setText("")
        self.label_13.setPixmap(QtGui.QPixmap("record.png"))
        self.label_13.setScaledContents(True)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_13.addWidget(self.label_13)
        self.timer_2 = QtWidgets.QLCDNumber(self.horizontalLayoutWidget_12)
        self.timer_2.setMaximumSize(QtCore.QSize(70, 50))
        self.timer_2.setObjectName("timer_2")
        self.horizontalLayout_13.addWidget(self.timer_2)
        self.record2Button = QtWidgets.QPushButton(self.horizontalLayoutWidget_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.record2Button.sizePolicy().hasHeightForWidth())
        self.record2Button.setSizePolicy(sizePolicy)
        self.record2Button.setObjectName("record2Button")
        self.horizontalLayout_13.addWidget(self.record2Button)
        self.stop2Button = QtWidgets.QPushButton(self.horizontalLayoutWidget_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stop2Button.sizePolicy().hasHeightForWidth())
        self.stop2Button.setSizePolicy(sizePolicy)
        self.stop2Button.setObjectName("stop2Button")
        self.horizontalLayout_13.addWidget(self.stop2Button)
        self.choose2Button = QtWidgets.QPushButton(self.horizontalLayoutWidget_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.choose2Button.sizePolicy().hasHeightForWidth())
        self.choose2Button.setSizePolicy(sizePolicy)
        self.choose2Button.setObjectName("choose2Button")
        self.horizontalLayout_13.addWidget(self.choose2Button)
        self.progressBar = QtWidgets.QProgressBar(self.tab_2)
        self.progressBar.setGeometry(QtCore.QRect(10, 160, 521, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_14 = QtWidgets.QLabel(self.tab_2)
        self.label_14.setGeometry(QtCore.QRect(50, 190, 121, 101))
        self.label_14.setText("")
        self.label_14.setPixmap(QtGui.QPixmap("profile.png"))
        self.label_14.setScaledContents(True)
        self.label_14.setObjectName("label_14")
        self.photo_field = QtWidgets.QLineEdit(self.tab_2)
        self.photo_field.setGeometry(QtCore.QRect(50, 300, 121, 27))
        self.photo_field.setReadOnly(True)
        self.photo_field.setObjectName("photo_field")
        self.label_15 = QtWidgets.QLabel(self.tab_2)
        self.label_15.setGeometry(QtCore.QRect(10, 310, 531, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.horizontalLayoutWidget_13 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_13.setGeometry(QtCore.QRect(10, 340, 521, 51))
        self.horizontalLayoutWidget_13.setObjectName("horizontalLayoutWidget_13")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_13)
        self.horizontalLayout_14.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_14.setSpacing(6)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_16 = QtWidgets.QLabel(self.horizontalLayoutWidget_13)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_14.addWidget(self.label_16)
        self.file_path_2 = QtWidgets.QLineEdit(self.horizontalLayoutWidget_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_path_2.sizePolicy().hasHeightForWidth())
        self.file_path_2.setSizePolicy(sizePolicy)
        self.file_path_2.setReadOnly(True)
        self.file_path_2.setObjectName("file_path_2")
        self.horizontalLayout_14.addWidget(self.file_path_2)
        self.browseButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browseButton.sizePolicy().hasHeightForWidth())
        self.browseButton.setSizePolicy(sizePolicy)
        self.browseButton.setObjectName("browseButton")
        self.horizontalLayout_14.addWidget(self.browseButton)
        self.horizontalLayoutWidget_14 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_14.setGeometry(QtCore.QRect(10, 390, 521, 51))
        self.horizontalLayoutWidget_14.setObjectName("horizontalLayoutWidget_14")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_14)
        self.horizontalLayout_15.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_15.setSpacing(6)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.progressBar_2 = QtWidgets.QProgressBar(self.horizontalLayoutWidget_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar_2.sizePolicy().hasHeightForWidth())
        self.progressBar_2.setSizePolicy(sizePolicy)
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.horizontalLayout_15.addWidget(self.progressBar_2)
        self.startButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.startButton.sizePolicy().hasHeightForWidth())
        self.startButton.setSizePolicy(sizePolicy)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout_15.addWidget(self.startButton)
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setGeometry(QtCore.QRect(370, 180, 121, 151))
        self.label_17.setText("")
        self.label_17.setPixmap(QtGui.QPixmap("genie.png"))
        self.label_17.setScaledContents(True)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setGeometry(QtCore.QRect(180, 190, 191, 101))
        self.label_18.setText("")
        self.label_18.setPixmap(QtGui.QPixmap("chat1.png"))
        self.label_18.setScaledContents(True)
        self.label_18.setObjectName("label_18")
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setGeometry(QtCore.QRect(100, 80, 426, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_12.setFont(font)
        self.label_12.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_12.setObjectName("label_12")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayoutWidget_15 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_15.setGeometry(QtCore.QRect(10, 130, 521, 61))
        self.horizontalLayoutWidget_15.setObjectName("horizontalLayoutWidget_15")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_15)
        self.horizontalLayout_16.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_16.setSpacing(6)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_19 = QtWidgets.QLabel(self.horizontalLayoutWidget_15)
        self.label_19.setMaximumSize(QtCore.QSize(100, 100))
        self.label_19.setText("")
        self.label_19.setPixmap(QtGui.QPixmap("record.png"))
        self.label_19.setScaledContents(True)
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_16.addWidget(self.label_19)
        self.timer_3 = QtWidgets.QLCDNumber(self.horizontalLayoutWidget_15)
        self.timer_3.setMaximumSize(QtCore.QSize(100, 100))
        self.timer_3.setObjectName("timer_3")
        self.horizontalLayout_16.addWidget(self.timer_3)
        self.recordButton3 = QtWidgets.QPushButton(self.horizontalLayoutWidget_15)
        self.recordButton3.setObjectName("recordButton3")
        self.horizontalLayout_16.addWidget(self.recordButton3)
        self.stopButton3 = QtWidgets.QPushButton(self.horizontalLayoutWidget_15)
        self.stopButton3.setObjectName("stopButton3")
        self.horizontalLayout_16.addWidget(self.stopButton3)
        self.label_20 = QtWidgets.QLabel(self.tab_3)
        self.label_20.setGeometry(QtCore.QRect(370, 210, 121, 161))
        self.label_20.setText("")
        self.label_20.setPixmap(QtGui.QPixmap("genie.png"))
        self.label_20.setScaledContents(True)
        self.label_20.setObjectName("label_20")
        self.profile_pic = QtWidgets.QLabel(self.tab_3)
        self.profile_pic.setGeometry(QtCore.QRect(50, 210, 121, 101))
        self.profile_pic.setText("")
        self.profile_pic.setPixmap(QtGui.QPixmap("profile.png"))
        self.profile_pic.setScaledContents(True)
        self.profile_pic.setObjectName("profile_pic")
        self.profile_label = QtWidgets.QLineEdit(self.tab_3)
        self.profile_label.setGeometry(QtCore.QRect(50, 320, 121, 27))
        self.profile_label.setAlignment(QtCore.Qt.AlignCenter)
        self.profile_label.setReadOnly(True)
        self.profile_label.setObjectName("profile_label")
        self.label_22 = QtWidgets.QLabel(self.tab_3)
        self.label_22.setGeometry(QtCore.QRect(180, 210, 191, 101))
        self.label_22.setText("")
        self.label_22.setPixmap(QtGui.QPixmap("chat1.png"))
        self.label_22.setScaledContents(True)
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.tab_3)
        self.label_23.setGeometry(QtCore.QRect(10, 80, 531, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.resetButton = QtWidgets.QPushButton(self.tab_3)
        self.resetButton.setGeometry(QtCore.QRect(220, 370, 99, 27))
        self.resetButton.setObjectName("resetButton")
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 550, 25))
        self.menuBar.setObjectName("menuBar")
        self.menuVoice_Recognition = QtWidgets.QMenu(self.menuBar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.menuVoice_Recognition.sizePolicy().hasHeightForWidth())
        self.menuVoice_Recognition.setSizePolicy(sizePolicy)
        self.menuVoice_Recognition.setObjectName("menuVoice_Recognition")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.menuBar.addAction(self.menuVoice_Recognition.menuAction())
        self.stopButton.setEnabled(False)
        self.stop2Button.setEnabled(False)
        self.stopButton3.setEnabled(False)
        self.updateButton.setEnabled(False)
        self.trainButton.setEnabled(False)
        self.quitButton.clicked.connect(QApplication.instance().quit)
        self.recordButton.clicked.connect(self.start_recording)
        self.stopButton.clicked.connect(self.stop_recording)
        self.chooseButton.clicked.connect(self.choose_sample)
        self.uploadButton.clicked.connect(self.choose_photo)
        self.saveButton.clicked.connect(self.save_profile)
        self.clearButton.clicked.connect(self.clear_info)
        self.updateButton.clicked.connect(self.update_info)
        self.trainButton.clicked.connect(self.start_training)
        self.record2Button.clicked.connect(self.start_recording2)
        self.stop2Button.clicked.connect(self.stop_recording2)
        self.choose2Button.clicked.connect(self.choose_sample2)
        self.checkBox.stateChanged.connect(self.state_changed)
        self.newTaskButton.clicked.connect(self.clear_2)
        self.recordButton3.clicked.connect(self.start_recording3)
        self.stopButton3.clicked.connect(self.stop_recording3)
        self.resetButton.clicked.connect(self.clear3)
        self.center(MainWindow)
        MainWindow.setWindowIcon(QIcon('web.png')) 
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)      
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.recordButton.setText(_translate("MainWindow", "Record"))
        self.stopButton.setText(_translate("MainWindow", "Stop"))
        self.label.setText(_translate("MainWindow", "Or"))
        self.chooseButton.setText(_translate("MainWindow", "Choose file"))
        self.label_3.setText(_translate("MainWindow", "Register Voice"))
        self.saveButton.setText(_translate("MainWindow", "Save"))
        self.trainButton.setText(_translate("MainWindow", "Train"))
        self.quitButton.setText(_translate("MainWindow", "Quit"))
        self.uploadButton.setText(_translate("MainWindow", "Upload Photo"))
        self.label_4.setText(_translate("MainWindow", "User Details"))
        self.label_6.setText(_translate("MainWindow", "Switch User"))
        self.label_7.setText(_translate("MainWindow", "Name :"))
        self.label_9.setText(_translate("MainWindow", "Sex :"))
        self.sex_select.setItemText(0, _translate("MainWindow", "Male"))
        self.sex_select.setItemText(1, _translate("MainWindow", "Female"))
        self.label_8.setText(_translate("MainWindow", "Age :"))
        self.clearButton.setText(_translate("MainWindow", "Clear Info"))
        self.updateButton.setText(_translate("MainWindow", "Update Info"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Registration"))
        self.label_11.setText(_translate("MainWindow", "Recognition"))
        self.newTaskButton.setText(_translate("MainWindow", "New Task"))
        self.checkBox.setText(_translate("MainWindow", "Get identity:"))
        self.record2Button.setText(_translate("MainWindow", "Record"))
        self.stop2Button.setText(_translate("MainWindow", "Stop"))
        self.choose2Button.setText(_translate("MainWindow", "Choose File"))
        self.label_15.setText(_translate("MainWindow", "Multiple tasks from file"))
        self.label_16.setText(_translate("MainWindow", "Input folder:"))
        self.browseButton.setText(_translate("MainWindow", "Browse"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.label_12.setText(_translate("MainWindow", "*Tick and we will verify your identity"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Recognition"))
        self.recordButton3.setText(_translate("MainWindow", "Record"))
        self.stopButton3.setText(_translate("MainWindow", "Stop"))
        self.label_23.setText(_translate("MainWindow", "Conversation Mode"))
        self.resetButton.setText(_translate("MainWindow", "Reset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Conversation"))
        self.menuVoice_Recognition.setTitle(_translate("MainWindow", "Voice Recognition"))

    def center(self, MainWindow):
        qr = MainWindow.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        MainWindow.move(qr.topLeft())

    def selectionchange(self,i):
        name = self.switch_user.currentText()
        if(name != "[New User]"):
            self.saveButton.setEnabled(False)
            self.updateButton.setEnabled(True)
            self.trainButton.setEnabled(True)
        else:
            self.clear_info()
            self.saveButton.setEnabled(True)
            self.updateButton.setEnabled(False)
        cursor = conn.execute("SELECT NAME, AGE, SEX, PROFILE, VOICE from USERS WHERE NAME='"+name+"'")
        for row in cursor:
            self.name_edit.setText(row[0])
            if row[2] == 'Male':
                self.sex_select.setCurrentIndex(0)
            else:
                self.sex_select.setCurrentIndex(1)
            self.age_select.setValue(row[1])
            self.imagepath = [""]
            self.imagepath[0] = row[3]
            self.label_5.setPixmap(QtGui.QPixmap(row[3]))
            self.file_path_1.setText(row[4])
            self.filepath = [""]
            self.filepath[0] = row[4]

    def display_lcd_1(self):
        d = 0
        while self.recording:
            t.sleep(1)
            self.timer.display(d)
            d = d+1

    def display_lcd_2(self):
        d = 0
        while self.recording2:
            t.sleep(1)
            self.timer_2.display(d)
            d = d+1

    def display_lcd_3(self):
        d = 0
        while self.recording3:
            t.sleep(1)
            self.timer_3.display(d)
            d = d+1

    def start_recording(self):
        self.stopButton.setEnabled(True)
        self.recordButton.setEnabled(False)
        self.background_thread = Thread(target=self.record_audio)
        self.background_thread.start()
        self.background_thread = Thread(target=self.display_lcd_1)
        self.background_thread.start()

    def start_recording2(self):
        self.stop2Button.setEnabled(True)
        self.record2Button.setEnabled(False)
        self.background_thread = Thread(target=self.record_audio2)
        self.background_thread.start()
        self.background_thread = Thread(target=self.display_lcd_2)
        self.background_thread.start()

    def start_recording3(self):
        self.stopButton3.setEnabled(True)
        self.recordButton3.setEnabled(False)
        self.recording3 = True
        self.background_thread = Thread(target=self.record_audio3)
        self.background_thread.start()
        self.background_thread = Thread(target=self.display_lcd_3)
        self.background_thread.start()

    def record_audio(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=au_format, channels=no_channels,
            rate=sampling_rate, input=True,
            frames_per_buffer=chunk)
        self.recording = True
        print ("Recording...")
        self.frames = []
        while self.recording: 
            try:
                data = self.stream.read(chunk)
            except OSError as ose:
                print ('Stopped..')
            self.frames.append(data)

    def record_audio2(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=au_format, channels=no_channels,
            rate=sampling_rate, input=True,
            frames_per_buffer=chunk)
        self.recording2 = True
        print ("Recording...")
        self.frames = []
        while self.recording2: 
            try:
                data = self.stream.read(chunk)
            except OSError as ose:
                print ('Stopped..')
            self.frames.append(data)

    def record_audio3(self):
        while self.recording3:
            print('Recording...')
            myrecording = sd.rec(int(sample_duration * sampling_rate), samplerate=sampling_rate, channels=1, blocking=True)
            print(myrecording.shape)
            #print (myrecording)
            #print (type(myrecording))
            self.background_thread2 = Thread(target=self.detect_stream(myrecording))
            self.background_thread2.start()

    def detect_stream(self,myrecording):
        '''
        repr = obj.generate_representation_array(myrecording)
        #print(repr.shape)
        repr1 = repr
        self.cursor1 = conn.execute("SELECT NAME,REP FROM USERS")
        score_list = []
        score_l = []
        for row in self.cursor1:
            repr = [float(i) for i in row[1].split(',')]
            score = np.dot(repr1,repr)
            score_l.append(score)
            temp_list = [row[0],score]
            score_list.append(temp_list)
        print (score_list)
        score_l.sort(reverse=True)
        for (name,s) in score_list:
            if (s == score_l[0]):
                self.display2(name)
                print ('Current speaker: '+name)
        '''
        name = obj2.conversation_mode(sampling_rate,myrecording)
        if name == 'unknown':
            self.display3()
        else:
            self.display2(name)

    def display2(self,name):
        self.cursor2 = conn.execute("SELECT NAME, PROFILE from USERS WHERE NAME='"+name+"'")
        for row in self.cursor2:
            self.profile_label.setText(row[0])
            self.profile_label.setAlignment(Qt.AlignCenter)
            self.profile_pic.setPixmap(QtGui.QPixmap(row[1]))

    def display3(self):
        self.profile_label.setText('unknown')
        self.profile_label.setAlignment(Qt.AlignCenter)
        self.profile_pic.setPixmap(QtGui.QPixmap('profile.png'))

    def stop_recording(self):
        self.recording = False
        self.stopButton.setEnabled(False)
        self.recordButton.setEnabled(True)
        print ("Finished recording.\n")
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.file_name = 'file_'+str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")+'.wav'
            path_name = os.getcwd()
            self.filepath = [""]
            self.filepath[0] = path_name+"/"+self.file_name
            self.file_path_1.setText(str(self.filepath[0]))
            self.waveFile = wave.open(self.file_name, 'wb')
            self.waveFile.setnchannels(no_channels)
            self.waveFile.setsampwidth(self.audio.get_sample_size(au_format))
            self.waveFile.setframerate(sampling_rate)
            self.waveFile.writeframes(b''.join(self.frames))
            self.waveFile.close()
            self.stream = None
        except Exception as e:
            print('Stream closed')

    def stop_recording2(self):
        self.recording2 = False
        self.stop2Button.setEnabled(False)
        self.record2Button.setEnabled(True)
        print ("Finished recording.\n")
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.file_name = 'file_'+str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")+'.wav'
            path_name = os.getcwd()
            self.filepath2 = [""]
            self.filepath2[0] = path_name+"/"+self.file_name
            self.waveFile = wave.open(self.file_name, 'wb')
            self.waveFile.setnchannels(no_channels)
            self.waveFile.setsampwidth(self.audio.get_sample_size(au_format))
            self.waveFile.setframerate(sampling_rate)
            self.waveFile.writeframes(b''.join(self.frames))
            self.waveFile.close()
            self.stream = None
        except Exception as e:
            print('Stream closed')

    def stop_recording3(self):
        self.recording3 = False
        self.stopButton3.setEnabled(False)
        self.recordButton3.setEnabled(True)
        print ("Finished recording.\n")

    def choose_photo(self):
      self.imagepath = QFileDialog.getOpenFileName(MainWindow, 'Open file', 
         './',"*.jpeg")
      try:
          if not str(self.imagepath[0]) == "":
              print('Image selected: '+str(self.imagepath[0]))
              self.label_5.setPixmap(QPixmap(str(self.imagepath[0])))
      except NameError as ne:
          print ('Canceled.')

    def choose_sample(self):
      self.filepath = QFileDialog.getOpenFileName(MainWindow, 'Open file', 
         './',"*.wav")
      try:
          print('Wav file selected: '+self.filepath[0])
          self.file_path_1.setText(str(self.filepath[0]))
      except NameError as ne:
          print("Canceled.")

    def choose_sample2(self):
      self.filepath2 = QFileDialog.getOpenFileName(MainWindow, 'Open file', 
         './',"*.wav")
      try:
          print('Wav file selected: '+self.filepath2[0])
      except NameError as ne:
          print("Canceled.")

    def save_profile(self):
        try:
            name = self.name_edit.text()
            name = name.strip()
            sex = str(self.sex_select.currentText())
            age = self.age_select.value()
            voice = self.filepath[0]
            image = self.imagepath[0]
            if(name == None):
                print ('Enter name first.')
            else:
                cursor.execute("SELECT * FROM USERS")
                results = cursor.fetchall()
                data = int(len(results))
                data = data + 1
                sql = '''INSERT INTO USERS (ID,NAME,AGE,SEX,VOICE,PROFILE) VALUES(?, ?, ?, ?, ?, ?);'''
                conn.execute(sql,[str(data),name,str(age),sex,voice,image]) 
                conn.commit()
                print (name + " " + sex + " "+ str(age)+ " "+voice+" "+ image)
                self.saveButton.setEnabled(False)
                self.updateButton.setEnabled(True)
                self.trainButton.setEnabled(True)
                self.update_entries()
                self.showdialog('Entry saved.','New entry has been created with provided details.')
        except Exception as e:
            self.showdialog('Enter details first.','Please make sure you have entered all the fields.')
            print ('Enter details first.')

    def update_entries(self):
        name = self.name_edit.text()
        name = name.strip()
        self.switch_user.clear()
        cursor = conn.execute("SELECT NAME from USERS")
        alist = ["[New User]"]
        for row in cursor:
            alist.extend(row)
        self.switch_user.addItems(alist)
        index = self.switch_user.findText(name)
        self.switch_user.setCurrentIndex(index)

    def update_info(self):
        try:
            name = self.name_edit.text()
            name = name.strip()
            sex = str(self.sex_select.currentText())
            age = self.age_select.value()
            voice = self.filepath[0]
            image = self.imagepath[0]
            if(name == None):
                print ('Enter name first.')
            else:
                name = self.switch_user.currentText()
                cursor = conn.execute("SELECT ID FROM USERS WHERE NAME='"+name+"'")
                index = None
                for row in cursor:
                    index = str(row[0])
                    print (index)
                sql = '''UPDATE USERS SET NAME = ? ,SEX = ? ,AGE = ? ,VOICE = ? ,PROFILE = ? WHERE ID = ?'''
                conn.execute(sql, [name,sex,str(age),voice,image,index])
                conn.commit()
                print ('Updated: '+name + " " + sex + " "+ str(age)+ " "+voice+" "+ image)
                self.update_entries()
                self.showdialog('Information updated.','Provides fields has been save to database.')
        except Exception as e:
            self.showdialog('Enter details first.','Please make sure you have entered all the fields.')
            print ('Enter details first.')

    def clear_info(self):
        try:
            self.name_edit.clear()
            self.sex_select.setCurrentIndex(0)
            self.age_select.setValue(1)
            self.label_5.setPixmap(QtGui.QPixmap("profile.png"))
            self.file_path_1.setText("")
        except Exception as e:
            self.showdialog('Enter details first.','Please make sure you have entered all the fields.')
            print ('Enter details first.')

    def clear_2(self):
        self.filepath2 = None
        self.timer_2.display(0)
        self.checkBox.setChecked(False)
        self.identity_field.setText("")
        self.photo_field.setText("")
        self.label_14.setPixmap(QtGui.QPixmap("profile.png"))
        self.label_18.setPixmap(QtGui.QPixmap("chat1.png"))
        self.progressBar.setValue(0)

    def clear3(self):
        self.timer_3.display(0)
        self.profile_label.setText("")
        self.profile_pic.setPixmap(QtGui.QPixmap("profile.png"))
        self.label_22.setPixmap(QtGui.QPixmap("chat1.png"))

    def showdialog(self,msgText,detailText):
       msg = QMessageBox()
       msg.setIcon(QMessageBox.Information)
       msg.setText(msgText)
       msg.setInformativeText(detailText)
       msg.setWindowTitle("Alert")
       msg.setStandardButtons(QMessageBox.Ok)
       retval = msg.exec_()

    def start_training(self):
        name = self.name_edit.text()
        name = name.strip()
        print(self.filepath[0])
        '''
        y, sr = self.read_file_for_noise_cancelation(self.filepath[0])
        print ("Audio time series: "+ str(y) + "\nSampling rate: " + str(sr))
        y_reduced_centroid_s = self.reduce_noise_centroid_simple(y, sr)
        y_reduced_centroid_s, time_trimmed = self.trim_silence(y_reduced_centroid_s)
        print ("Silent seconds trimmed: " + str(time_trimmed))
        self.output_file('./noise_cleaned_samples/',name+'_noise_cleaned_simple.wav', y_reduced_centroid_s, sr, '_ctr_s')
        #y_reduced_centroid_mb = self.reduce_noise_centroid_bass_boost(y, sr)
        #y_reduced_centroid_mb, time_trimmed = self.trim_silence(y_reduced_centroid_mb)
        #print ("Silent seconds trimmed: " + str(time_trimmed))
        #self.output_file('noise_cleaned_samples/' ,filename, y_reduced_centroid_mb, sr, '_ctr_bb')
        repr = obj.generate_representation('./noise_cleaned_samples/'+name+'_noise_cleaned_simple_ctr_s.wav')
        '''
        repr = obj.generate_representation(self.filepath[0])
        print("success")
        print(repr.shape)
        try:
            name = self.name_edit.text()
            name = name.strip()
            if(name == None):
                print ('Enter name first.')
            else:
                name = self.switch_user.currentText()
                cursor = conn.execute("SELECT ID FROM USERS WHERE NAME='"+name+"'")
                index = None
                for row in cursor:
                    index = str(row[0])
                    print (index)
                y = ",".join(str(i) for i in repr)
                sql = '''UPDATE USERS SET REP = ? WHERE ID = ?'''
                conn.execute(sql, [y,index])
                conn.commit()
                print ('Updated: representation for '+name)
                self.update_entries()
                self.showdialog('Information updated.','Provides fields has been save to database.')
        except Exception as e:
            self.showdialog('Enter details first.','Please make sure you have entered all the fields.')
            print ('Enter details first.')

    def state_changed(self):
       # try:
            if(self.checkBox.isChecked()):
                self.value = 0
                self.background_thread = Thread(target=self.display_progress)
                self.background_thread.start()
                print('Starting identification for '+self.filepath2[0])
                y, sr = self.read_file_for_noise_cancelation(self.filepath2[0])
                print ("Audio time series: "+ str(y) + "\nSampling rate: " + str(sr))
                y_reduced_centroid_s = self.reduce_noise_centroid_simple(y, sr)
                y_reduced_centroid_s, time_trimmed = self.trim_silence(y_reduced_centroid_s)
                print ("Silent seconds trimmed: " + str(time_trimmed))
                self.output_file('./noise_cleaned_samples/','temp_noise_cleaned_simple.wav', y_reduced_centroid_s, sr, '_ctr_s')
                cursor = conn.execute("SELECT NAME,REP FROM USERS")
                score_list = []
                score_l = []
                length = len(cursor.fetchall())
                cursor = conn.execute("SELECT NAME,REP FROM USERS")
                iteration = 1
                for row in cursor:
                    repr = [float(i) for i in row[1].split(',')]
                    score = obj.similarity_score(repr,'./noise_cleaned_samples/temp_noise_cleaned_simple_ctr_s.wav')
                    self.value = (100/length)*iteration
                    iteration = iteration + 1
                    score_l.append(score)
                    temp_list = [row[0],score]
                    score_list.append(temp_list)
                print (score_list)
                score_l.sort(reverse=True)
                for (name,s) in score_list:
                    if (s == score_l[0]):
                        self.display(name)
                        print (name)
                    
       # except Exception as e:
        #    self.checkBox.setChecked(False)
         #   self.showdialog('Enter details first.','Please make sure you have entered all the fields.')
          #  print ('Enter details first.')

    def display_progress(self):
        while self.value < 100:
            t.sleep(0.1)
            self.progressBar.setValue(self.value)
        self.progressBar.setValue(100)

    def display(self,name):
        cursor = conn.execute("SELECT NAME, PROFILE from USERS WHERE NAME='"+name+"'")
        for row in cursor:
            self.identity_field.setText(row[0])
            self.photo_field.setText(row[0])
            self.photo_field.setAlignment(Qt.AlignCenter)
            self.label_14.setPixmap(QtGui.QPixmap(row[1]))
            self.label_18.setPixmap(QtGui.QPixmap("chat2.png"))

    def play_audio(self,path):
        wf = wave.open(path, 'rb')
        audio = pyaudio.PyAudio()
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True)
        data = wf.readframes(chunk)
        while data != '':
            stream.write(data)
            data = wf.readframes(chunk)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print ("\n")

    def read_file_for_noise_cancelation(self,path):
        y, sr = librosa.load(path,sr=None)
        return y, sr

    def reduce_noise_centroid_simple(self,y, sr):
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        threshold_h = np.max(cent)
        threshold_l = np.min(cent)
        less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5).limiter(gain=6.0)
        y_cleaned = less_noise(y)
        return y_cleaned

    def reduce_noise_centroid_bass_boost(self,y, sr):
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        threshold_h = np.max(cent)
        threshold_l = np.min(cent)
        less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5).highshelf(gain=-30.0, frequency=threshold_h, slope=0.5).limiter(gain=10.0)
        #less_noise = AudioEffectsChain().lowpass(frequency=threshold_h).highpass(frequency=threshold_l)
        y_cleaned = less_noise(y)
        cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
        columns, rows = cent_cleaned.shape
        boost_h = math.floor(rows/3*2)
        boost_l = math.floor(rows/6)
        boost = math.floor(rows/3)
        #boost_bass = AudioEffectsChain().lowshelf(gain=20.0, frequency=boost, slope=0.8)
        boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)#.lowshelf(gain=-20.0, frequency=boost_l, slope=0.8)
        y_clean_boosted = boost_bass(y_cleaned)
        return y_clean_boosted

    def trim_silence(self,y):
        y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
        trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)
        return y_trimmed, trimmed_length

    def mfcc_extract(self,path):
        (rate,sig) = wav.read(path)
        mfcc_feat = mfcc(sig,rate)
        fbank_feat = logfbank(sig,rate)
        print(fbank_feat[1:3,:])

    def output_file(self,destination ,filename, y, sr, ext=""):
        destination = destination + filename[:-4] + ext + '.wav'
        librosa.output.write_wav(destination, y, sr)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle('Speaker Recognition')
    MainWindow.show()
    sys.exit(app.exec_())

