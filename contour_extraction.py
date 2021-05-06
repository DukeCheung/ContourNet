import cv2
import os

if __name__ == '__main__':
    img = cv2.imread('../../3dsMax/[2,-10,1.7].jpg')
    # canny = cv2.Canny(img, 150, 150)
    # cv2.imshow('Canny', canny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for root, sub_dir, files in os.walk(r'D:/SYSU/3dsMax'):
        i = 0
        for file in files:
            print(file.split('.')[-1])
            img = cv2.imread(os.path.join(root, file))
            print(img.shape)
            img = cv2.resize(img, (224, 224))
            canny = cv2.Canny(img, 150, 150)
            print(canny.shape)
            print(type(canny))
            # canny = cv2.cvtColor(canny, cv2.COLOR_RGB2GRAY)
            # cv2.imwrite('D:/SYSU/output/{}.jpg'.format(i), canny)
            print(canny.shape)
            i = i+1
            # cv2.imshow('Canny', canny)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(os.path.join(root, file))
