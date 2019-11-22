# ProjectImageprocessing
# เป็นการเรียกใช้การทำของ Image Segmentation เเต่ยังไม่สมบูรณ์โดยเเบ่งหลักการให้ส่งข้อมูลเป็นรูปภาพที่มีร่างกายเป็นมนุษย์เเละใช้ opencv-python ในการตรวจจับเป็นภาพ Skeleton ขึ้นมา ยังมีปัญหาติดขัดอยู่ครับ เนื่องจากเพิ่งได้ศึกษาการเขียน Opencv-python ส่วนใหญ่จะใช้ Video Capture ในการตรวจจับวัตถุเเละเเสดงออกมาในลักษณะร่างของ Skeleton 

# วิธีใช้งาน Coding
# ลง Module ให้เรียบร้อย
# python -m pip install --user opencv-contrib-python
# pip install opencv-python
# ใช้คำสั่งในการ Run Coding
# python skeletonrgb.py --input test1.jpg 
# ปล.ในไฟล์มีภาพทั้งหมด 6 ภาพสามาถใช้ได้หมด
# --input เป็นกำหนดค่าเริ่มต้นที่ต้องใช้งานรูปภาพนั้น
# --thr เป็นการกำหนด threshold ของ input ภาพเพื่อนำมาตรวจสอบ


# ภาพ Original ที่เเสดงต้องเเก้ใน โค้ดทุกครั้งเวลาจะทำการรันโค้ดนี้ >>> im = cv2.imread("test5.jpg") # ภาพ .jpg ไว้เทส 
