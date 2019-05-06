import numpy as np
import cv2

kernel=cv2.getGaussianKernel(256,50) + 0.0
kernel=kernel*kernel.T
#scales all the values and make the center vaule of kernel to be 1.0
kernel=kernel/np.max(kernel)
#print(kernel)

heatmap=kernel*255
heatmap=heatmap.astype(np.uint8)
#heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
cv2.imshow('heatmap',heatmap)
cv2.waitKey(0)
