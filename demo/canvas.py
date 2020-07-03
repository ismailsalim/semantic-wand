from PIL import Image, ImageTk, ImageDraw
import math

class CanvasImage:
    def __init__(self, canvas_frame, canvas):
        self.img_frame = canvas_frame
        self.canvas = canvas
        self.container = None
        self.img = None
        self.annotations = None


    def reload_img(self, img):
        self.canvas.delete('all') # clear annotations
        
        self.img = img
        self.imwidth, self.imheight = img.size
        
        # reset annotations
        self.last_x = None
        self.last_y = None
        self.annotations = Image.new('L', (self.imwidth, self.imheight))

        self.show_img()  

        self.canvas.focus_set() 


    def show_img(self):
        imagetk = ImageTk.PhotoImage(self.img)
        imageid = self.canvas.create_image(0, 0,anchor='nw', image=imagetk)                                

        self.canvas.lower(imageid)  # set image into background
        self.canvas.imagetk = imagetk  # extra reference for garbage collection

        self.drawing = ImageDraw.Draw(self.annotations)
        self.canvas.bind('<1>', self.activate_paint)


    def activate_paint(self, e):
        self.canvas.bind('<B1-Motion>', self.paint)
        self.last_x, self.last_y = e.x, e.y


    def paint(self, e):
        x, y = e.x, e.y   
        self.canvas.create_line((self.last_x, self.last_y, x, y), width=1) 
        self.drawing.line((self.last_x, self.last_y, x, y), fill=255, width=1)
        self.last_x, self.last_y = x, y