import cv2 as cv
import imutils

import cv2



def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # Находим градиент по оси X
    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Находим градиент в направлении y
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # Преобразуем значение градиента в 8 бит
    x_grad = cv.convertScaleAbs(grad_x)
    y_grad = cv.convertScaleAbs(grad_y)

    cv.imshow("x_grad", x_grad)
    cv.imshow("y_grad", y_grad)
    # Объединить два градиента
    src1 = cv.addWeighted(x_grad, 0.3, y_grad, 0.3, 0)
    cv.imshow("src1", src1)
    # Объедините градиенты, используя хитрый алгоритм, где 50 и 100 - пороги
    edge = cv.Canny(src1, 50, 100)
    cv.imshow("Canny_edge_1", edge)
    edge1 = cv.Canny(grad_x, grad_y, 10, 100)
    cv2.imwrite("edge1.jpg", edge1)
    cv.imshow("Canny_edge_2", edge1)
    # Используйте край как маску для выполнения побитовых и побитовых операций
    edge2 = cv.bitwise_and(image, image, mask=edge1)
    cv.imshow("bitwise_and", edge2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edge1, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("closed1.jpg", closed)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # s=cv2.drawContours(image.copy(), cnts, -1, (0, 255, 0), 4)
    print(len(cnts))
    cnts = imutils.grab_contours(cnts)
    total = 0
    # цикл по контурам
    for c in cnts:
        # аппроксимируем (сглаживаем) контур
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        # если у контура 4 вершины, предполагаем, что это книга
        if len(approx) >= 4  :
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
            total += 1
    # показываем результирующее изображение
    print(" нашлось  {0} ".format(total))
    cv2.imwrite("output.jpg", image)


src = cv.imread("232323.jpg")
cv.imshow("2", src)
edge_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()

