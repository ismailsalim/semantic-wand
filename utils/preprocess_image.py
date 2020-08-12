import cv2

def rescale_img(img, max_img_dim, interpolation):
    h, w = img.shape[:2]

    if h < max_img_dim and w < max_img_dim:
        return img

    if h > w:
        r = max_img_dim/float(h)
        dim = (int(w*r), max_img_dim)
    else:
        r = max_img_dim/float(w)
        dim = (max_img_dim, int(h*r))

    return cv2.resize(img, dim, interpolation=interpolation)


def preprocess_scribbles(scribbles, img=None):
    if img is not None:
        assert img.shape[:2] == scribbles.shape[:2], (
            "Image: {} and Scribbles: {} must be same shape!".format(img.shape[:2], scribbles.shape[:2]))


    scribbles[scribbles == 128] = -1 # convert unnannotated areas
    scribbles[scribbles == 255] = 1 # convert fg scribbles
    scribbles[scribbles == 0] = 0 # convert bg scribbles

    return scribbles