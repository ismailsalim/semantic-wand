import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import os
from PIL import Image
import numpy as np

from demo.controller import Controller
from demo.canvas import CanvasImage

class App(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.root.title('INTERACTIVE DEMO')
        self.root.geometry("1200x1000")
        self.root.resizable(width=False, height=False)

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
        self.canvas_frame = tk.LabelFrame(self.root, width=800, height=800)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, 
                                width=800, height=800)
        self.canvas.grid(row=0, column=0, sticky='NSW', padx=5, pady=5)

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

        # keep name for saving matte later
        img = cv2.imread(filename.name)
        self.controller.filename, _ = os.path.splitext(os.path.basename(filename.name))
        
        # resize image to the fixed canvas size 
        max_img_dim = min(self.canvas.winfo_width(), self.canvas.winfo_height())
        rescaled_img = self.rescale_img(img, max_img_dim)
        
        self.controller.set_img(rescaled_img)
    

    def rescale_img(self, img, max_img_dim):
        (h, w) = img.shape[:2]

        if h > w:
            r = max_img_dim/float(h)
            dim = (int(w*r), max_img_dim)
        else:
            r = self.max_img_dim/float(w)
            dim = (max_img_dim, int(h*r))

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    def save_matte(self):
        self.menu.focus_set()
        if self.controller.matte is None:
            return

        filename = filedialog.asksaveasfilename(parent=self.root, 
                                            initialfile='{}_matte'.format(self.controller.filename),
                                            filetypes = [('PNG image', '*.png')],
                                            title = 'Save mask as...')
        cv2.imwrite('{}.png'.format(filename), self.controller.matte)


    def process_img(self):
        if self.img_on_canvas:
            # self.img_on_canvas.annotations.save('test.png')
            self.controller.process_img()
        

    def update_canvas(self, img, matte=False):
        if self.img_on_canvas is None:
            self.img_on_canvas = CanvasImage(self.canvas_frame, self.canvas)

        if not matte: # display input image
            self.img_on_canvas.reload_img(Image.fromarray(img))
        else: # display matte
            matte = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2RGBA))
            self.img_on_canvas.reload_img(matte)



