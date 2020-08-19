from demo.controller import Controller
from demo.canvas import CanvasImage

from pipeline.masking_stage import MaskingStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage
from pipeline.pipe import Pipeline, NoInstancesFoundError, NoSubjectFoundError
from utils.preprocess_image import rescale_img

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
from PIL import Image
import numpy as np

class App(tk.Frame):
    def __init__(self, root, pipeline, max_img_dim=800):
        super().__init__(root)
        self.root = root
        self.root.title("Semantic Wand")
        self.root.geometry("1200x1000")
        self.root.resizable(width=True, height=True)

        self.controller = Controller(pipeline, self.update_canvas)

        self.max_img_dim = max_img_dim

        self.add_menu()
        self.add_canvas()
        self.add_workspace()

        self.active_brush_button = None


    def add_menu(self):
        self.menu = tk.LabelFrame(self.root, bd=1)
        self.menu.pack(side=tk.TOP, fill="x")
        self.load_button = tk.Button(self.menu, text="Load", command=self.load_img)
        self.load_button.pack(side=tk.LEFT)

        self.save_scribbles = tk.Button(self.menu, text="Save Scribbles", 
                                    command=lambda: self.save(np.array(self.img_on_canvas.scribbles),
                                                                        "_scribbles"))
        self.save_scribbles.pack(side=tk.LEFT)

        self.save_instances = tk.Button(self.menu, text="Save Instances", 
                                    command=lambda: self.save(self.controller.instances,
                                                            "_instances"))
        self.save_instances.pack(side=tk.LEFT)

        self.save_heatmap = tk.Button(self.menu, text="Save Heatmap", 
                                    command=lambda: self.save(self.controller.heatmap,
                                                                "_heatmap"))
        self.save_heatmap.pack(side=tk.LEFT)

        self.save_trimap = tk.Button(self.menu, text="Save Trimap", 
                                    command=lambda: self.save(self.controller.trimap,
                                                                "_trimap"))
        self.save_trimap.pack(side=tk.LEFT)

        self.save_alpha = tk.Button(self.menu, text="Save Alpha", 
                                    command=lambda: self.save(self.controller.alpha,
                                                                "_alpha"))
        self.save_alpha.pack(side=tk.LEFT)

        self.save_matte = tk.Button(self.menu, text="Save Matte", 
                                    command=lambda: self.save(self.controller.matte,
                                                                "_matte"))
        self.save_matte.pack(side=tk.LEFT)

        self.quit_button = tk.Button(self.menu, text="Quit", command=self.root.quit)
        self.quit_button.pack(side=tk.LEFT)


    def add_canvas(self):
        self.canvas_frame = tk.LabelFrame(self.root, width=self.max_img_dim, height=self.max_img_dim)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, 
                                width=self.max_img_dim, height=self.max_img_dim)
        self.canvas.grid(row=0, column=0, sticky="NSW", padx=5, pady=5)

        self.img_on_canvas = None

        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    
    def add_workspace(self):
        self.workspace = tk.LabelFrame(self.root, bd=1, text="Workspace")
        self.workspace.pack(side=tk.TOP, fill="x")

        self.process_button = tk.Button(self.workspace, text="Extract Object", command=self.process_img)
        self.process_button.pack(side=tk.TOP, pady=20)

        self.fg_brush_button = tk.Button(self.workspace, text="Foreground Brush", 
                                        command=lambda: self.activate_brush(self.fg_brush_button, "FG_BRUSH"))
        self.fg_brush_button.pack(side=tk.TOP, pady=10)

        self.bg_brush_button = tk.Button(self.workspace, text="Background Brush",
                                        command=lambda: self.activate_brush(self.bg_brush_button, "BG_BRUSH"))
        self.bg_brush_button.pack(side=tk.TOP, pady=10)

        self.brush_size_slider = tk.Scale(self.workspace, from_=1, to=25, orient=tk.HORIZONTAL, label="Size")
        self.brush_size_slider.pack(side=tk.TOP, pady=10)

        self.reset_button = tk.Button(self.workspace, text="Reset scribbles", command=self.reset_scribbles)
        self.reset_button.pack(side=tk.TOP, pady= 10)


    def load_img(self):
        self.menu.focus_set()
        filename = filedialog.askopenfile(parent=self.root, 
            title = "Select image",
            filetypes=[("Images", "*.jpg *.JPG *.jpeg *.png")])

        img = cv2.imread(filename.name)
        self.controller.filename, _ = os.path.splitext(os.path.basename(filename.name))
        
        img = rescale_img(img, self.max_img_dim, cv2.INTER_AREA)
        
        self.controller.set_img(img)


    def save(self, img, img_type):
        self.menu.focus_set()
        if img is not None:
            filename = filedialog.asksaveasfilename(parent=self.root, 
                                                initialfile=self.controller.filename + img_type,
                                                filetypes = [("PNG image", "*.png")], title = "Save as...")
            cv2.imwrite(filename, img)


    def process_img(self):
        if self.img_on_canvas:
            scribbles = self.img_on_canvas.scribbles

            try:
                self.controller.process_img(scribbles)
            except NoSubjectFoundError:
                tk.messagebox.showerror("No subject found!", "See instances output!")
            except NoInstancesFoundError:
                tk.messagebox.showerror("No Instances Found!", "Try another image!")



    def activate_brush(self, button_pressed, brush_type):
        if self.active_brush_button is not None:
            self.active_brush_button.config(relief=tk.RAISED)
        
        if self.active_brush_button == button_pressed:
            self.active_brush_button = None
            self.img_on_canvas.active_brush = None
        else:
            self.active_brush_button = button_pressed
            self.active_brush_button.config(relief=tk.SUNKEN)
            if self.img_on_canvas:
                self.img_on_canvas.active_brush = brush_type


    def reset_scribbles(self):
        if self.img_on_canvas:
            self.img_on_canvas.reload_img()


    def update_canvas(self, img, with_matte=False):
        if self.img_on_canvas is None:
            self.img_on_canvas = CanvasImage(self.canvas_frame, self.canvas, self.brush_size_slider)

        if with_matte: # display matte results
            matte = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2RGBA))
            self.img_on_canvas.reload_img(matte)
        else: # display loaded input image
            self.img_on_canvas.reload_img(Image.fromarray(img))




