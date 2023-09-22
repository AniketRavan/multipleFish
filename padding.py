class padding:
    def __call__(self, image):
        w, h = image.shape
        w_buffer = 141 - w
        w_left = int(w_buffer/2)
        w_right = w_buffer - w_left
        w_buffer = 141 - h
        w_top = int(w_buffer/2)
        w_bottom = w_buffer - w_top
        padding = (w_left, w_top, w_right, w_bottom)
        pad_transform = transforms.Pad(padding)
        padded_image = pad_transform(image)
        return padded_image
