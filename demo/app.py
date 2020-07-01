import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import os
from PIL import Image
import numpy as np

from demo.controller import Controller
from demo.canvas import CanvasImage

class App(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        root.title('INTERACTIVE DEMO')

        self.controller = Controller(self.update_canvas)

        self.add_menu()
        self.add_canvas()
        self.add_workspace()


    def add_menu(self):
        self.menu = tk.LabelFrame(self.root, bd=1)
        self.menu.pack(side=tk.TOP, fill='x')
        load_button = tk.Button(self.menu, text='Load', command=self.load_img)
        load_button.pack(side=tk.LEFT)
        save_button = tk.Button(self.menu, text='Save', command=self.save_matte)
        save_button.pack(side=tk.LEFT)
        quit_button = tk.Button(self.menu, text='Quit', command=self.root.quit)
        quit_button.pack(side=tk.LEFT)


    def add_canvas(self):
        self.canvas_frame = tk.LabelFrame(self.root)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, 
                                width=1000, height=800)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.img_on_canvas = None

        self.canvas_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

    
    def add_workspace(self):
        self.workspace = tk.LabelFrame(self.root, bd=1, text='Workspace')
        self.workspace.pack(side=tk.TOP, fill='x')
        process_button = tk.Button(self.workspace, text='Process', command=self.process_img)
        process_button.pack(side=tk.LEFT)


    def load_img(self):
        self.menu.focus_set()
        filename = filedialog.askopenfile(parent=self.root, 
            title = 'Select image',
            filetypes=[('Images', '*.jpg *.JPG *.jpeg *.png')])

        img = cv2.imread(filename.name)
        self.controller.filename, _ = os.path.splitext(os.path.basename(filename.name))

        self.controller.set_img(img)
    

    def save_matte(self):
        self.menu.focus_set()
        if self.controller.matte is None:
            return

        filename = filedialog.asksaveasfile(parent=self.root, 
                                            initialfile='{}.png'.format(self.controller.filename),
                                            filetypes = [('PNG image', '*.png')],
                                            title = 'Save mask as...')


    def process_img(self):
        if self.img_on_canvas:
            self.controller.process_img()
        

    def update_canvas(self, img, matte=False):
        if self.img_on_canvas is None:
            self.img_on_canvas = CanvasImage(self.canvas_frame, self.canvas)

        if matte == False:
            self.img_on_canvas.reload_img(Image.fromarray(img))
        else:
            m = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2RGBA))
            self.img_on_canvas.reload_img(m)



