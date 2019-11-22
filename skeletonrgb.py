import cv2 as cv
import numpy as np
import argparse #เป็น module ที่กำหนด argument ที่จะเรียกใช้งานได้ทาง console
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser() #ตัวกำหนดเรียกใช้ที่ console
parser.add_argument('--input') #เพิ่มรูปภาพโดยใช้ argument สั่งที่ console
parser.add_argument('--thr', default=0.2, type=float) #ปรับระดับค่า threshold 
parser.add_argument('--width', default=368, type=int) #กำหนดความกว้างของ figure
parser.add_argument('--height', default=368, type=int) #กำหนดความสูงของ figure
img = cv.imread("3.jpg", 0) # ภาพ 3,4 .jpg ไว้เทส
args = parser.parse_args() #ตัวกำหนดเรียกใช้ argument ทั้งหมด

#กำหนดส่วนลักษณะต่างๆของร่างกายมนุษย์
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

#กำหนดท่าทางให้คล้องกับร่างกายมนุษย์
POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ] 


#เเสดงหน้าต่าง gui
inWidth = args.width 
inHeight = args.height
#เรียกใช้ไฟล์ graph_opt.pb
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

#ai ตรวจจับรูปภาพที่เข้ามาทาง input
cap = cv.VideoCapture(args.input if args.input else 0)

#หลังจากตรวจจับเเละผ่านการตรวจสอบจะเข้า loop whille เพื่อทำไฟล์รูปภาพขึ้นมาให้ชื่อว่า frame ถ้าหากไม่ผ่านก็อาจจะ error หรือรอการตรวจสอบต่อไป
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    #เเสดงรูปร่างที่ได้ในหน้าต่าง gui
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    #เป็น output ที่เรียกใช้ 19 องค์ประกอบของร่างกายมนุษย์
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  
    #assert เป็นฟังก์ชันที่ไว้เทสว่าองค์ประกอบร่างกายของมนุษย์ตรงตาม output ที่เช็คเข้ามาไหม
    assert(len(BODY_PARTS) == out.shape[1])
    
    #กำหนดตัวเเปร points ขึ้นมาค่าว่างปล่าว
    points = []
    
    #คำสั่งวนซํ้าให้ค้นหาชิ้นส่วนของร่างกายให้ตรงตามองค์ประกอบ 19 ส่วน
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :] #ตรวจสอบชิ้นส่วนร่างกายที่มีความสอดคล้องกัน
       
        #กำหนดตัวเเปรเช็คค่าสูงสุดเเละค่าตํ่าสุดของ input
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        
        #จุดที่มีการเช็คค่า ถ้าหากค่าของ threshold มีค่าที่สูงกว่าปกติ    
        points.append((int(x), int(y)) if conf > args.thr else None)
    
    #คำสั่ง loop ที่จะค้นหาองค์ประกอบให้ครบสมบูรณ์ที่สุด
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
  
        #กำหนดตัวเเปรเพิ่มเติม
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        #ใช้คำสั่ง if เพื่อตรวจสอบข้อมูลของรูปภาพที่เข้ามาเเละเช็คเเต่ละเฟรม
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    
    #กำหนดค่าหน้าต่างของ figure เเบบใน matlab
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()
    skeleton = skeletonize(frame)
    
    #เเสดงรูปภาพที่ถูกประมวลผล
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('original')
    ax[0].axis('off')

    ax[1].imshow(frame, cmap=plt.cm.gray)
    ax[1].set_title('skeleton')
    ax[1].axis('off')

    #คำนวณเวลาในการประมวลผลรูปภาพ
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    print('Success')
    #cv.imshow('Human Skeleton using OpenCV', frame) #ตัวเเสดงทั้งกล้องเเละ gui

    fig.tight_layout()
    plt.show()