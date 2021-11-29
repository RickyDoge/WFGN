import cv2



class HarrAdaBoostModel(object):
    def __init__(self, xml_dir):
        self.face_detector = cv2.CascadeClassifier(xml_dir)

    def show(self, fdir):
        img = cv2.imread(fdir)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_img, 1.3, 3)
        print('face = ', len(faces))
        print(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('dat', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    model = HarrAdaBoostModel(r'D:\Project\PyCharmProjects\ImagePlay\weight\haarcascade_frontalface_default.xml')
    model.show(r'D:\Training Dataset\FurGen\2\other (35).jpeg')

