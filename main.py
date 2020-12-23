import cv2


def auto_face_detect():
    face_dc = cv2.CascadeClassifier('data.xml')
    cap = cv2.VideoCapture(0)
    while True:
        _, get_images = cap.read()
        color_mode = cv2.cvtColor(get_images, cv2.COLOR_BGR2RGB)
        get_face = face_dc.detectMultiScale(color_mode, 1.2, 4)
        for (x_axis, y_axis, width, height) in get_face:
            cv2.rectangle(get_images, (x_axis, y_axis), (x_axis + width, y_axis + height), (255, 0, 0), 2)
        cv2.imshow('Your Face', get_images)
        key_response = cv2.waitKey(30) & 0xff
        if key_response == 27:
            break
    cap.release()


if __name__ == '__main__':
    auto_face_detect()
