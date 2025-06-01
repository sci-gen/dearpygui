import numpy as np
import cv2


def create_sample_image():
    """Create a sample image (RGBA format)"""
    width, height = 200, 200

    # RGBA形式で直接作成
    image_data = []
    for i in range(height):
        for j in range(width):
            # 正規化された値（0.0-1.0）
            r = i / height
            g = j / width
            b = 0.5
            a = 1.0  # アルファチャンネル

            image_data.extend([r, g, b, a])

    return image_data


if __name__ == "__main__":
    sample_image = create_sample_image()
    print("Sample image created with data length:", len(sample_image))
    cv2.imwrite(
        "sample_image.png", np.array(sample_image).reshape(200, 200, 4) * 255
    )  # Save as PNG for verification
