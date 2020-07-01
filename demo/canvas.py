from PIL import Image, ImageTk
import math

class CanvasImage:
    def __init__(self, canvas_frame, canvas):
        self.current_scale = 1.0
        self.delta = 1.2
        
        self.imgframe = canvas_frame
        self.canvas = canvas
        
        self.container = None
    

    def reload_img(self, img):
        self.original_img = img.copy()
        self.current_img = img.copy()

        self.imwidth, self.imheight = self.original_img.size
        self.min_side = min(self.imwidth, self.imheight)  

        scale = min(self.canvas.winfo_width() / self.imwidth, 
                    self.canvas.winfo_height() / self.imheight)
        if self.container:
            self.canvas.delete(self.container)

        self.container = self.canvas.create_rectangle(
            (0, 0, scale * self.imwidth, scale * self.imheight), width=0)
        
        self.current_scale = scale

        self.show_img()  
        self.canvas.focus_set() 
        

    def show_img(self):
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            border_width = 2
            sx1, sx2 = x1 / self.current_scale, x2 / self.current_scale
            sy1, sy2 = y1 / self.current_scale, y2 / self.current_scale
            crop_x, crop_y = max(0, math.floor(sx1 - border_width)), max(0, math.floor(sy1 - border_width))
            crop_w, crop_h = math.ceil(sx2 - sx1 + 2 * border_width), math.ceil(sy2 - sy1 + 2 * border_width)
            crop_w = min(crop_w, self.original_img.width - crop_x)
            crop_h = min(crop_h, self.original_img.height - crop_y)

            current_img = self.original_img.crop((crop_x, crop_y,
                                                  crop_x + crop_w, crop_y + crop_h))
            crop_zw = int(round(crop_w * self.current_scale))
            crop_zh = int(round(crop_h * self.current_scale))
            zoom_sx, zoom_sy = crop_zw / crop_w, crop_zh / crop_h
            crop_zx, crop_zy = crop_x * zoom_sx, crop_y * zoom_sy
            self.real_scale = (zoom_sx, zoom_sy)

            interpolation = Image.NEAREST if self.current_scale > 2.0 else Image.ANTIALIAS
            current_img = current_img.resize((crop_zw, crop_zh), interpolation)
            zx1, zy1 = x1 - crop_zx, y1 - crop_zy
            zx2 = min(zx1 + self.canvas.winfo_width(), current_img.width)
            zy2 = min(zy1 + self.canvas.winfo_height(), current_img.height)

            self.current_img = current_img.crop((zx1, zy1, zx2, zy2))

            imagetk = ImageTk.PhotoImage(self.current_img)
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection