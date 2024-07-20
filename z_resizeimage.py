from PIL import Image

# 打开图像
image = Image.open("n03991062_126.JPEG")

# 调整图像大小
resized_image = image.resize((256, 256))

# 保存调整大小后的图像
resized_image.save("output1.jpg")

image = Image.open("clip_0.jpg")

# 调整图像大小
resized_image = image.resize((256, 256))

# 保存调整大小后的图像
resized_image.save("output2.jpg")