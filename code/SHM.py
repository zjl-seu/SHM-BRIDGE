# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:38:01 2021

@author: zjl-seu
"""
import cv2
import time
import numpy as np
import tkinter as tk
from VRmodel import VR
from threading import Thread
from PIL import Image, ImageTk
from tkinter import scrolledtext
from PIL import Image, ImageDraw, ImageFont
            
class GUI():
    def __init__(self, width=1300, height=650):
        self.w = width
        self.h = height
        self.title = "桥梁车载识别系统"
        self.root = tk.Tk(className=self.title)
        self.root.iconbitmap('tk.ico')
        
        self.list1 = tk.StringVar()
        self.list2 = tk.StringVar()
        self.list3 = tk.StringVar()
        self.list1.set("test.mp4")
        #页面一
        self.page0 = tk.Frame(self.root)
        self.photo = tk.PhotoImage(file='桥梁背景.png')
        tk.Label(self.page0, text="桥梁车载时空分布识别系统", justify="center", image=self.photo, compound="center", font=("华文行楷", 50), fg="blue").place(x=0, y=0, width=1300, height=600)
        #text_label.pack()
        #tk.Label(self.page0, font=('楷体', 50), text="桥梁车载时空分布识别系统").place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.page0.pack(fill=tk.BOTH, expand=True)
        
        #页面二
        self.page1 = tk.Frame(self.root)
        self.frame1_1 = tk.Frame(self.page1)
        self.frame1_2 = tk.Frame(self.page1)
        self.frame1_2_1 = tk.Frame(self.frame1_2)
        self.frame1_2_2 = tk.Frame(self.frame1_2) 
        self.frame1_2_2_1 = tk.Frame(self.frame1_2_2) 
        self.frame1_2_2_2 = tk.Frame(self.frame1_2_2) 
        
        label1_1 = tk.Label(self.frame1_1, text="")
        label1_2 = tk.Label(self.frame1_1, font=('楷体', 25), text="桥梁上高清摄像机显示与识别") 
        label1_3 = tk.Label(self.frame1_1, text="")
        label1_4 = tk.Label(self.frame1_2_1, font=('楷体',15), text="拍摄画面")
        self.canvas1_1 = tk.Canvas(self.frame1_2_1, width=800, height=500, bg="#c4c2c2")
        label1_5 = tk.Label(self.frame1_2_2_1, font=('楷体',15), text="请输入视频地址：")
        entry1_1 = tk.Entry(self.frame1_2_2_1, textvariable=self.list1, highlightcolor="Fuchsia", highlightthickness=1, width=50)
        label1_6 = tk.Label(self.frame1_2_2_1, text="")
        label1_7 = tk.Label(self.frame1_2_2_1, font=('楷体',15), text="识别结果")     
        self.scrolledtext1_1 = scrolledtext.ScrolledText(self.frame1_2_2_1, font=('楷体',10), width=50, height=27, wrap=tk.WORD)
        label1_8 = tk.Label(self.frame1_2_2_1, text="")
        button1_1 = tk.Button(self.frame1_2_2_2, text="打开", font=('楷体',15), fg='Purple', width=15, height=2, command=self.video_open)
        label1_9 = tk.Label(self.frame1_2_2_2, text="  ")
        button1_2 = tk.Button(self.frame1_2_2_2, text="识别", font=('楷体',15), fg='Purple', width=15, height=2, command=self.detect_stop)
        label1_10 = tk.Label(self.frame1_2_2_2, text="  ")
        button1_3 = tk.Button(self.frame1_2_2_2, text="停止", font=('楷体',15), fg='Purple', width=15, height=2, command=self.video_close)
        
        self.frame1_1.pack()
        self.frame1_2.pack()
        self.frame1_2_1.grid(row=0, column=0)
        self.frame1_2_2.grid(row=0, column=1) 
        self.frame1_2_2_1.grid(row=0, column=0)
        self.frame1_2_2_2.grid(row=1, column=0) 
        
        label1_1.grid(row=0, column=0)
        label1_2.grid(row=1, column=0)
        label1_3.grid(row=2, column=0)
        label1_4.grid(row=0, column=0)
        self.canvas1_1.grid(row=1, column=0)
        label1_5.grid(row=0, column=0)
        entry1_1.grid(row=1, column=0)
        label1_6.grid(row=2, column=0)
        label1_7.grid(row=3, column=0)
        self.scrolledtext1_1.grid(row=4, column=0)  
        label1_8.grid(row=5, column=0)
        button1_1.grid(row=0, column=0) 
        label1_9.grid(row=0, column=1)
        button1_2.grid(row=0, column=2)
        label1_10.grid(row=0, column=3)
        button1_3.grid(row=0, column=4)
        
        self.page1.forget()
        #页面三
        self.page2 = tk.Frame(self.root, bg='red')
        self.page2.forget()
        
        self.create_page() 
        self.vr = VR()
        
    def create_page(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        filemenu1 = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='车牌识别展示', menu=filemenu1)
        filemenu1.add_command(label='开始', command=self.page1_show)
        filemenu2 = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='车流结果展示', menu=filemenu2)
        filemenu2.add_command(label='开始', command=self.page2_show)  
        
    def page1_show(self):
        self.page0.forget()
        self.page1.pack(fill=tk.BOTH, expand=True)
        self.page2.forget()
             
    def page2_show(self):
        self.page0.forget()
        self.page1.forget()
        self.page2.pack(fill=tk.BOTH, expand=True)
               
    @staticmethod
    def thread_it(func, *args):
        t = Thread(target=func, args=args) 
        t.setDaemon(True)   # 守护--就算主界面关闭，线程也会留守后台运行（不对!）
        t.start()           # 启动
     
    def video_open(self):
        self.scrolledtext1_1.delete(0.0, "end")
        self.list2.set("1")
        self.list3.set("0")
        video = self.list1.get()
        t1 = time.time()
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            if self.list2.get() == "1":
                ret, frame = cap.read()
                if ret == True:
                    self.video_play(frame)
                    if self.list3.get() == "1":
                        self.thread_it(self.video_detect, frame)
                else:
                    break
            else:
                break
        
        cap.release()  
        print(time.time()-t1)  
        
    def plt_rec(self, image):
        fontc = ImageFont.truetype("simsun.ttc", 20, encoding="unic")
        cv2.rectangle(image, (int(333), int(147)), (int(542), int(392)), (0, 0, 255), 2, cv2.LINE_AA)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((int(333)+10, int(147)-30), "ROI(500*500)", (0, 0, 255), font=fontc)
        imagex = np.array(image) 
        return imagex
        
    def video_play(self, frame):
        img = cv2.resize(frame, (800, 500), interpolation=cv2.INTER_CUBIC) 
        img = self.plt_rec(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        photo = ImageTk.PhotoImage(image=img)
        self.canvas1_1.create_image([400, 250], image=photo)
        self.canvas1_1.update_idletasks()
        self.canvas1_1.update()      
            
    def video_detect(self, frame):
        image = frame[200 : 700, 700 : 1200]
        result = self.vr.main(image)
        if result != None:
            print(result[1])
            self.scrolledtext1_1.insert("insert", result[1]+"-"+str(300)+'\n')

    def detect_stop(self):
        self.list3.set("1")
        
    def video_close(self):
        self.list2.set("0")
    
    def center(self):
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = int((ws/2) - (self.w/2))
        y = int((hs/2) - (self.h/2))
        self.root.geometry("{}x{}+{}+{}".format(self.w, self.h, x, y))
    
    #监听到关闭窗体的后，弹出提示信息框，提示是否真的要关闭，若是的话，则关闭 
    def window_close_handle(self):
        if tk.messagebox.askyesnocancel("关闭确认窗口","确认要关闭窗口吗？"):
            self.root.destroy()#关闭窗口
        
    #函数说明：loop等待用户事件
    def loop(self):
        #禁止修改窗口大小
        self.root.resizable(False, False)
        #窗口居中
        self.center()
        self.root.protocol('WM_DELETE_WINDOW', self.window_close_handle) 
        self.root.mainloop()

if __name__ == "__main__":
    gui = GUI()
    gui.loop()
