import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

i=0
while(cap.isOpened()):
    flag,frame=cap.read()
    if flag==False:
        break
    path='E:/face recognization/faces/face'+str(i)+'.jpg'
    cv2.imwrite(path,frame)
    i+=1

cap.release()
cv2.destroyAllWindows()