import cv2

# face classifier
face_detector_file = cv2.CascadeClassifier('face_Detect.xml')
smile_detector_file = cv2.CascadeClassifier('smile_Detect.xml')

# grab web cam video
myVideo = cv2.VideoCapture(0)

# show the current frame
while True:
    successful_frame_read, frame = myVideo.read()
    if not successful_frame_read:
        break
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_detector_file.detectMultiScale(frame_grayscale)

    # draw rectangles around faces
    for (x, y, w, h) in faces:
        the_face = frame[y:y+h, x:x+w]

        # detect smile in the face area
        smiles = smile_detector_file.detectMultiScale(
            the_face, scaleFactor=1.7, minNeighbors=10)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=1,
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, color=(255, 255, 255))

        # drawing rectangle around smiles
        # for (x_, y_, w_, h_) in smiles:
        #     cv2.rectangle(the_face, (x_, y_),
        #                   (x_+w_, y_+h_), (100, 200, 50), 2)

    cv2.imshow("Hey Buddy", frame)
    H = cv2.waitKey(1)

myVideo.release()
cv2.destroyAllWindows()
print("Python:- Aman You've forgotten me buddy😔")
