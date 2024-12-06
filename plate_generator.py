import enum
import cv2
import random
import numpy as np
import random_plate
import black_plate
import blue_plate
import green_plate
import yellow_plate
from math import sin, cos


class PlateType(enum.Enum):
    BLACK = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    RANDOM = 4


plate_type_obj_map = {
    PlateType.BLACK: black_plate.Draw(),
    PlateType.BLUE: blue_plate.Draw(),
    PlateType.GREEN: green_plate.Draw(),
    PlateType.YELLOW: yellow_plate.Draw(),
    PlateType.RANDOM: random_plate.Draw()
}


class Draw:
    _draw = [
        black_plate.Draw(),
        blue_plate.Draw(),
        yellow_plate.Draw(),
        green_plate.Draw()
    ]
    _provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
    _alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                  "W", "X", "Y", "Z"]
    _ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
            "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    _candidates = [_provinces, _alphabets]

    def __init__(self, plate_type: PlateType = PlateType.RANDOM):
        self.plate_type = plate_type

    def draw_green_plate(self):
        candidates = self._candidates + [self._ads] * 6
        label = "".join([random.choice(c) for c in candidates])
        return green_plate.Draw()(label, random.randint(0, 1)), label

    def draw_black_plate(self):
        if random.random() < 0.5:
            candidates = self._candidates + [self._ads] * 4
            candidates += [["港", "澳"]]
        else:
            candidates = self._candidates + [self._ads] * 5
        label = "".join([random.choice(c) for c in candidates])
        return black_plate.Draw()(label), label

    def draw_yellow_plate(self):
        if random.random() < 0.5:
            candidates = self._candidates + [self._ads] * 4
            candidates += [["学"]]
        else:
            candidates = self._candidates + [self._ads] * 5
        label = "".join([random.choice(c) for c in candidates])
        return yellow_plate.Draw()(label), label

    def draw_blue_plate(self):
        candidates = self._candidates + [self._ads] * 5
        label = "".join([random.choice(c) for c in candidates])
        return blue_plate.Draw()(label), label

    def __call__(self):
        if self.plate_type == PlateType.GREEN:
            return self.draw_green_plate()
        elif self.plate_type == PlateType.BLUE:
            return self.draw_blue_plate()
        elif self.plate_type == PlateType.YELLOW:
            return self.draw_yellow_plate()
        elif self.plate_type == PlateType.BLACK:
            return self.draw_black_plate()
        else:
            all_draw = [self.draw_green_plate, self.draw_blue_plate, self.draw_yellow_plate, self.draw_black_plate]
            return random.choice(all_draw)()


def r(val):
    return int(np.random.random() * val)


class TransFormationBase:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class AffineTransformation(TransFormationBase):
    def __init__(self, max_angel):
        super().__init__()
        self.max_angel = max_angel

    def __call__(self, img: np.ndarray):
        angel = r(self.max_angel * 2) - self.max_angel
        size_o = [img.shape[1], img.shape[0]]
        shape = img.shape
        size = (shape[1] + int(shape[0] * cos((float(self.max_angel) / 180) * 3.14)), shape[0])
        interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))
        pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
        if angel > 0:
            pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
        m = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, m, size)
        return dst


class PerspectiveTransformation(TransFormationBase):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def __call__(self, img: np.ndarray):
        """
        添加透视畸变
        """
        shape = [img.shape[1], img.shape[0]]
        pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        pts2 = np.float32([[r(self.factor), r(self.factor)], [r(self.factor), shape[0] - r(self.factor)],
                           [shape[1] - r(self.factor), r(self.factor)],
                           [shape[1] - r(self.factor), shape[0] - r(self.factor)]])
        m = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, m, shape)
        return dst


class Saturation(TransFormationBase):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        """
        添加饱和度光照的噪声
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
        hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
        hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img


class GaussBlur(TransFormationBase):
    def __init__(self, level):
        super().__init__()
        self.level = (level * 2 + 1, level * 2 + 1)

    def __call__(self, image):
        return cv2.blur(image, self.level)


class GaussNoise(TransFormationBase):
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        for i in range(image.shape[2]):
            c = image[:, :, i]
            diff = 255 - c.max()
            noise = np.random.normal(0, random.randint(1, 3), c.shape)
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            noise = diff * noise
            image[:, :, i] = c + noise.astype(np.uint8)
        return image


class Smudge(TransFormationBase):
    def __init__(self, smu_path):
        super().__init__()
        self.smudge_image = cv2.imread(smu_path)

    def __call__(self, plate_image: np.ndarray):
        y = random.randint(0, self.smudge_image.shape[0] - plate_image.shape[0])
        x = random.randint(0, self.smudge_image.shape[1] - plate_image.shape[1])
        texture = self.smudge_image[y:y + plate_image.shape[0], x:x + plate_image.shape[1]]
        return cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(plate_image), texture))


def fake_plate(plate_type: PlateType = PlateType.RANDOM, transforms: list[TransFormationBase] = ()):
    draw = Draw(plate_type)
    plate, label = draw()
    for transform in transforms:
        plate = transform(plate)
    return np.array(plate), label
