import cv2

img_1 = cv2.imread("images/0001.jpg")
# img_1 = cv2.resize(img_1, (1280, 720))
img_2 = cv2.imread("images/0010.jpg")
# img_2 = cv2.resize(img_2, (1280, 720))

# Convert it to grayscale
img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints_1, descriptors_1 = orb.detectAndCompute(img_1_gray, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(img_2_gray, None)

image_with_keypoints = cv2.drawKeypoints(img_1_gray, keypoints_1, None, flags=0)

cv2.imshow("ORB KeyPoints", image_with_keypoints)
cv2.waitKey(3000)
cv2.destroyAllWindows()

matcher = cv2.BFMatcher()
matches = matcher.match(descriptors_1, descriptors_2)

final_img = cv2.drawMatches(
    img_1,
    keypoints_1,
    img_2,
    keypoints_2,
    matches[:20],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# final_img = cv2.resize(final_img, (1000,650))

cv2.imshow("Matches", final_img)
cv2.waitKey(0)
