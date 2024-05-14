import os
import cv2

# Read the images
images = sorted(os.listdir('images'))
images_with_keypoints = []
keypoints_array = []
descriptors_array = []
for image in images:
    # Detect the keypoints and compute the descriptors with ORB
    img = cv2.imread('images/'+image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    keypoints_array.append(keypoints)
    descriptors_array.append(descriptors)
    # Draw the keypoints
    image_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0),flags=0)
    images_with_keypoints.append(image_with_keypoints)

# Save sequence of images as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('features.mp4', fourcc, 30.0, (1920, 1080))

# for image in images_with_keypoints:
#     out.write(image)
# out.release()
# cv2.destroyAllWindows()

# Keypoints matchings videofeed
matcher = cv2.BFMatcher()
matched_images = []
for i in range(len(images)):
    matches = matcher.match(descriptors_array[i], descriptors_array[i])
    final_img = cv2.drawMatches(images_with_keypoints[i], keypoints_array[i],
                                images_with_keypoints[i], keypoints_array[i], matches[:25], None)
    # matchColor=(100,255,100), singlePointColor=(0,255,0))
    matched_images.append(final_img)

out = cv2.VideoWriter('matched_points.mp4', fourcc, 30.0, (2*1920, 1080))
for image in matched_images:
    out.write(image)
out.release()
