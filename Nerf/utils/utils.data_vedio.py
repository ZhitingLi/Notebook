import os
import cv2 # type: ignore

"""处理视频数据，首先从视频中抽帧"""
def extract_images(video_path, output_folder):
    """获取视频文件名和创建输出文件夹"""
    # 获取视频文件名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 新建文件夹
    output_path = os.path.join(output_folder, video_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    """打开视频文件并设置帧间隔"""
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    # 设置帧间隔
    frame_interval = int(2)

    """逐帧提取并保存图像帧"""
    # 逐帧提取并保存满足间隔要求的帧
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(frame_interval)
            if count % frame_interval == 0:
                image_name = os.path.join(output_path, f"{video_name}_{count//frame_interval}.jpg")
                cv2.imwrite(image_name, frame)
            count += 1
        else:
            break
    cap.release()


if __name__ == '__main__':
    video_path = 'C:/Users/lizhi/Desktop/Deep_learning_zhitingli/nerf_byself/nerf_1' # 视频文件路径
    output_folder = 'C:/Users/lizhi/Desktop/Deep_learning_zhitingli/nerf_byself/nerf_1/test_frame'  # 输出文件夹路径
    extract_images(video_path, output_folder)