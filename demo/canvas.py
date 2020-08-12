from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import math
import numpy as np
import cv2

class CanvasImage:
    def __init__(self, canvas_frame, canvas, brush_size):
        self.img_frame = canvas_frame
        self.canvas = canvas
        self.container = None
        self.img = None
        self.scribbles = None
        self.active_brush = None
        self.brush_size = brush_size


    def reload_img(self, img=None):
        self.canvas.delete('all') # free scribbles 
        
        if img is not None:
            self.img = img
            self.imwidth, self.imheight = img.size

        self.last_x, self.last_y = None, None
        self.scribbles = np.ones((self.imheight, self.imwidth))*128

        self.show_img()  

        self.canvas.focus_set() 


    def show_img(self):
        imagetk = ImageTk.PhotoImage(self.img)
        imageid = self.canvas.create_image(0, 0,anchor='nw', image=imagetk)                                

        self.canvas.lower(imageid)  # set image into background
        self.canvas.imagetk = imagetk 

        self.canvas.bind('<1>', self.activate_paint)


    def activate_paint(self, e):
        self.canvas.bind('<B1-Motion>', self.paint)
        self.last_x, self.last_y = e.x, e.y


    def paint(self, e):
        x, y = e.x, e.y

        if e.x < self.imwidth and e.y < self.imheight:
            if self.active_brush == "FG_BRUSH":
                self.line = self.canvas.create_line((self.last_x, self.last_y, x, y), 
                                                    fill='blue', 
                                                    width=self.brush_size.get()*2, capstyle=tk.ROUND) 
                self.annot = cv2.circle(self.scribbles, (x,y) , self.brush_size.get(), 255, -1)
            elif self.active_brush == "BG_BRUSH":
                self.line = self.canvas.create_line((self.last_x, self.last_y, x, y), 
                                                    fill='red', 
                                                    width=self.brush_size.get()*2, capstyle=tk.ROUND)
                self.annot = cv2.circle(self.scribbles, (x,y) , self.brush_size.get(), 0, -1)

        self.last_x, self.last_y = x, y

    