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



import cv2
import numpy as np

def video_demo():
    capture = cv2.VideoCapture(0)  # 0为电脑内置摄像头
    photoname = 0  # 文件名序号初始值

    while (True):
        ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):  # #1s后执行，按s键截图保存
            photoname += 1
            filename = str(photoname) + '.png'  # filename为图像名字，将photoname作为编号命名保存的截图
            filepath = '/home/yuan/Mani-GPT/camera_capture' + '/' + filename
            cv2.imwrite(filepath, frame)  # 截图,filepath为保存路径,frame为此时的图像
            print(filename + '保存成功')  # 打印保存成功
            cv2.destroyAllWindows()
            capture.release()
            return filepath

        # if cv2.waitKey(1) & 0xFF == ord('q'):  #1s后执行，按q键退出
        #     break
        #     return True
        
    capture.release()







# import cv2

# def video_demo():
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 电脑自身摄像头
#     i = 0  # 定时装置初始值
#     photoname = 1  # 文件名序号初始值

#     while True:
#         i = i + 1
#         ret, frame = cap.read()
#         frame = cv2.flip(frame, 1)  # 图片左右调换
#         cv2.imshow('window', frame)

#         if i == 50:  # 定时装置，定时截屏，可以修改。
#             filename = str(photoname) + '.png'  # filename为图像名字，将photoname作为编号命名保存的截图
#             filepath = 'E:/asdfghj' + '\\' + filename
#             cv2.imwrite(filepath, frame)  # 截图 前面为放在桌面的路径 frame为此时的图像
#             print(filename + '保存成功')  # 打印保存成功
#             i = 0  # 清零

#             # 读取并显示图片
#             img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#             cv2.imshow('Read Image', img)

#             photoname = photoname + 1
#             if photoname >= 2:  # 最多截图20张 然后退出（如果调用photoname = 1 不用break为不断覆盖图片）
#                 break
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 释放资源
#     cap.release()

# video_demo()
# cv2.destroyAllWindows()



