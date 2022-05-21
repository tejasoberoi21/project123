import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import numpy as np 
import cv2
from PIL import Image
import PIL.ImageOps
import ssl, os

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
label = pd.read_csv("/Users/tejasoberoi/Downloads/datasets/C 122-123/labels.csv")["labels"]
pixels = np.load("/Users/tejasoberoi/Downloads/image.npz")["arr_0"]

classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
totalLabels = len(classes)

pixel_train,pixel_test,label_train,label_test=tts(pixels,label,random_state = 42, train_size = 7500,test_size = 2500)
pixel_train_scaled = pixel_train/255.0
pixel_test_scaled = pixel_test/255.0

lr = LogisticRegression(multi_class = "multinomial", random_state = 0).fit(pixel_train_scaled,label_train)
pred = lr.predict(pixel_test_scaled)
accuracy = accuracy_score(pred,label_test)
print(accuracy)

cv = cv2.VideoCapture(0)
while(True):
  try:
    ret,frame = cv.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    height,width = gray.shape
    uLeft = (int(width/2-56),int(height/2-56))
    bRight = (int(width/2+56),int(height/2+56))

    cv2.rectangle(gray,uLeft,bRight,(0,255,0), 2)
    roi = gray[uLeft[1]:bRight[1],uLeft[0]:bRight[0]]

    pilImg = Image.fromarray(roi)

    bw = pilImg.convert("L")

    bw = bw.resize((28,28),Image.ANTIALIAS)

    bw = PIL.ImageOps.invert(bw)

    minPixel = np.percentile(bw,20)

    scaledRes = np.clip(bw - minPixel,0,255)

    maxPixel = np.max(bw)

    scaledRes = np.asarray(scaledRes)/maxPixel

    finalRes = scaledRes/maxPixel 
    testSample = np.array(finalRes).reshape(1,784)

    #print(finalRes)
    test_pred = lr.predict(testSample)
    print("Predicted is: ",test_pred)
    cv2.imshow("Frame", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except:
    pass
  
cv.release()
cv2.destroyAllWindows()
print("finished")
