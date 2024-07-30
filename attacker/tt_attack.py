from PIL import Image
import numpy as np

class RandomAttacker():
    def __init__(self, data_batch):
        self.data_batch = data_batch

    def adjust_image(image_path, output_path):
        # 打开图片
        image = Image.open(image_path).convert('RGB')
        
        # 将图片转换为numpy数组
        image_np = np.array(image)
        
        # 获取图片的宽度和高度
        height, width, _ = image_np.shape
        
        for row in range(height):
            for col in range(width):
                # if col % 2 == 0 or row % 2 == 0:  # 偶数列或偶数行
                if col % 2 == 0:
                    # 将当前像素值转换为灰度值
                    r, g, b = image_np[row, col]
                    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                    image_np[row, col] = [gray, gray, gray]
        
        # 将numpy数组重新转换为PIL图片
        new_image = Image.fromarray(image_np.astype('uint8'), 'RGB')
        
        # 保存新图片
        new_image.save(output_path)
        
        return new_image
