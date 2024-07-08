# # import the opencv library
# import cv2

# # define a video capture object
# cap = cv2.VideoCapture(0)

# while True:

#     # capture the video frame by frame
#     ret, frame = cap.read()

#     # display the resulting frame
#     cv2.imshow('frame', frame)

#     # the 'q' button is set as the quitting button 
#     # you may use any desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # after the loop release the cap object
# cap.release()
# # destroy all the windows
# cv2.destroyAllWindows()




# import cv2
# import numpy as np

# def video_demo():
#     capture = cv2.VideoCapture(0)  # 0为电脑内置摄像头
#     while (True):
#         ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
#         frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
#         cv2.imshow("video", frame)
#         c = cv2.waitKey(1)  #1s后执行
#         if c == 27:  #按Esc键退出
#             break

# video_demo()
# cv2.destroyAllWindows()

import sys
import numpy as np
import pyzed.sl as sl
import cv2

import cv2
import numpy as np




help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

def point_cloud_format_name():     # 根据当前模式获取保存点云数据的文件扩展名
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 
  
def depth_format_name():     # 根据当前模式获取保存深度图数据的文件扩展名
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 

def save_point_cloud(zed, filename) :     # 从 ZED 相机获取点云数据, 将数据保存到带有相应扩展名的文件中, 打印保存是否成功的信息
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :     # 从 ZED 相机获取深度图数据, 将数据保存到带有相应扩展名的文件中, 打印保存是否成功的信息
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :     # 从 ZED 相机获取左图像和右图像, 水平拼接两个图像, 将拼接后的图像保存到文件中

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    # cv2.imwrite(filename, sbs_image)
    cv2.imwrite(filename, image_cv_left)




def process_key_event(zed, key) :
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68:
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
        return  path + prefix_depth + str(count_save)
    elif key == 110 or key == 78:
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
        return path + prefix_point_cloud + str(count_save)
    elif key == 109 or key == 77:
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1
        return "ZED_image" + str(count_save) + ".png"
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    # print(" Press 'm' to switch Point Cloud format")
    # print(" Press 'n' to switch Depth format")


def zed():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    key = ' '
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()

            cv2.imshow("Image", image_ocv)
            cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)

            filepath =process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")
    return filepath


# def video_demo():
#     capture = cv2.VideoCapture(0)  # 0为电脑内置摄像头
#     photoname = 0  # 文件名序号初始值

#     while (True):
#         ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
#         frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
#         cv2.imshow("video", frame)

#         if cv2.waitKey(1) & 0xFF == ord('s'):  # #1s后执行，按s键截图保存
#             photoname += 1
#             filename = str(photoname) + '.png'  # filename为图像名字，将photoname作为编号命名保存的截图
#             filepath = '/home/yuan/Mani-GPT/camera_capture' + '/' + filename
#             cv2.imwrite(filepath, frame)  # 截图,filepath为保存路径,frame为此时的图像
#             print(filename + '保存成功')  # 打印保存成功
#             cv2.destroyAllWindows()
#             capture.release()
#             return filepath

#        if cv2.waitKey(1) & 0xFF == ord('q'):  #1s后执行，按q键退出
#            break
#            return True
        
#    capture.release()




