from plate_generator import *
from PIL import Image

if __name__ == '__main__':
    transforms = [
        # Smudge("./res/smu.png"),
        # AffineTransformation(30),
        # PerspectiveTransformation(10),
        # Saturation(),
        # GaussBlur(random.randint(1, 4)),
        # GaussNoise()
    ]

    for i in range(20):
        plate, label = fake_plate(PlateType.RANDOM, transforms=transforms)
        image = Image.fromarray(plate)
        image.save(f"lp_image/{label}.png")
