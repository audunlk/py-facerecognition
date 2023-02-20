import pathlib
import cv2

#use cascade path C:\Users\audun\anaconda3\Lib\site-packages\cv2\data
cascade_path = pathlib.Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
print(cascade_path)

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
        

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()




