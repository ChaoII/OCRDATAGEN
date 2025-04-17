import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib.pyplot import imshow


def find_fitting_font_size(text, font_path, max_width, max_height, start_size=1, step=1):
    size = start_size
    while True:
        try:
            font = ImageFont.truetype(font_path, size)
            # Use getbbox to get the bounding box of the text
            bbox = font.getbbox(text)
            if bbox is None:
                # Some fonts might return None for certain characters, handle this case
                size -= step
                if size < start_size:
                    raise ValueError("Could not find a suitable font size")
                return ImageFont.truetype(font_path,
                                          size - step if size > start_size else start_size)  # Fallback to previous valid size
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if text_width > max_width or text_height > max_height:
                # If the text exceeds the dimensions, backtrack and return the previous size
                size -= step
                if size < start_size:
                    raise ValueError("Could not find a suitable font size")
                return ImageFont.truetype(font_path, size)
        except IOError:
            raise ValueError("Font file not found or could not be opened")
        size += step


class Draw:
    _font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "font/eng_92.ttf"), 126),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "font/zh_cn_92.ttf"), 95)
    ]
    _bg = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "res/yellow_220.png")), (440, 220))

    def __call__(self, plate):
        if len(plate) != 7:
            print("ERROR: Invalid length")
            return None
        fg = self._draw_fg(plate)
        return cv2.cvtColor(cv2.bitwise_or(fg, self._bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch, size, is_up=False):
        is_chinese = not (ch.isupper() or ch.isdigit()) or is_up
        font_path = "font/zh_cn_92.ttf" if is_chinese else "font/eng_92.ttf"
        font = ImageFont.truetype(font_path, 300)
        bbox = font.getbbox(ch)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if not is_chinese:
            text_width -= 30
        img = Image.new("RGB", (text_width, text_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        pos = (0 - bbox[0], 0 - bbox[1])
        draw.text(pos, ch, font=font, fill='white')
        cv_image = np.array(img)
        resize_image = cv2.resize(cv_image, size)
        return resize_image

    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (440, 220), (0, 0, 0)))
        img[15:75, 110:190] = self._draw_char(plate[0], (80, 60), True)
        img[15:75, 250:330] = self._draw_char(plate[1], (80, 60), True)
        start_gap = 27
        font_width = 65
        font_gap = 15
        for i in range(5):
            start_pos = start_gap + i * (font_width + font_gap)
            print(start_pos)
            img[90:200, start_pos:start_pos + font_width] = self._draw_char(plate[i + 2], (65, 110))
        cv2.imwrite("test.png", img)
        return img


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate a black plate.")
    parser.add_argument("plate", help="license plate number (default: 京A12345)", type=str, nargs="?",
                        default="京AF0236")
    args = parser.parse_args()

    draw = Draw()
    plate = draw(args.plate)
    plt.imshow(plate)
    plt.show()
