import cv2
import matplotlib.pyplot as plt


loc_pred = cv2.imread('predictions_val/predictions/test_localization_moore-tornado_00000063_1_prediction.png', 0)
loc_targ = cv2.imread('predictions_val/targets/test_localization_moore-tornado_00000063_1_target.png', 0)
dam_pred = cv2.imread('predictions_val/predictions/test_damage_moore-tornado_00000063_1_prediction.png', 0)
dam_targ = cv2.imread('predictions_val/targets/test_damage_moore-tornado_00000063_1_target.png', 0)

ax = plt.subplot(221)
ax.set_title('loc_pred')
plt.imshow(loc_pred)
ax = plt.subplot(222)
ax.set_title('loc_targ')
plt.imshow(loc_targ)
ax = plt.subplot(223)
ax.set_title('dam_pred')
plt.imshow(dam_pred)
ax = plt.subplot(224)
ax.set_title('dam_targ')
plt.imshow(dam_targ)
plt.show()

