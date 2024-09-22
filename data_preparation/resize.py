import os
import cv2

def resize_images(input_folder, output_folder, width, height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            # 构建文件路径
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # 调整图像大小
                resized_img = cv2.resize(img, (width, height))
                
                # 保存调整后的图像
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, resized_img)
                print(f'Resized and saved: {output_path}')
            else:
                print(f'Failed to load: {img_path}')

if __name__ == "__main__":
    # 定义输入输出文件夹路径
    input_folder = "/home/edge/work/datasets/PANDA_dataset/images/train"
    output_folder = "/home/edge/work/datasets/PANDA_dataset/images/train_resized_960x540"

    width = 960
    height = 540
    
    # 调用函数执行批量调整大小
    resize_images(input_folder, output_folder, width, height)
