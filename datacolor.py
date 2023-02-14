import cv2
from matplotlib import pyplot as plt

test_image = cv2.imread("mnist/result/misspreddata_16.png", 0) #グレースケール画像として読み込む.
test_image = cv2.bitwise_not(test_image)
plt.imshow(test_image, cmap = 'gray')

plt.savefig('misspreddata_16_grayscale.png')