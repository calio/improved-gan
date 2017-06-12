import tensorflow as tf
import model
from scipy.misc import imread, imresize

imgs = []
for i in range(1, 51):
    img = imread("/Users/calio/Downloads/imgs/img%d.jpg" % i)
    img = img[:100,:100,:]
    imgs.append(img)

scores = model.get_inception_score(imgs)

print(scores)
