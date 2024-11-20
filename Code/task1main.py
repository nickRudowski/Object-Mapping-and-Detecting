# Driver code for task 1

import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Constants
MIN_MATCH_COUNT_HIGH = 50
MIN_MATCH_COUNT_SUM = 140
RATIO_TEST_THRESHOLD = 0.70     
MEDIAN_KERNEL = 1


def resize_image(image, max_width=2048, max_height=2048):
    """Resize the image to fit within max dimensions while preserving aspect ratio."""
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv.resize(image, new_size, interpolation=cv.INTER_AREA)
    return image


def load_images(folder_path):
    """Load all images from a given folder."""
    images = []
    image_names = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            print("Loaded image: ", img_path)
            image = resize_image(cv.imread(img_path, cv.IMREAD_GRAYSCALE))
            assert image is not None, "error!!!!!!!!!!!"
            if image.dtype != "uint8":
                image = (255 * image).astype("uint8")
            cv.normalize(image, image, 0, 255, cv.NORM_MINMAX, dtype=None)
            image = cv.medianBlur(image, MEDIAN_KERNEL)
            images.append(image)
            image_names.append(filename)
    return images, image_names


def detect_and_compute(image, sift):
    """
    Detect keypoints and compute descriptors using SIFT.
    """
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(des1, des2, ratio=RATIO_TEST_THRESHOLD):
    """
    Match features using FLANN and apply Lowe's ratio test.
    """
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def find_homography_and_draw(img1, img2, kp1, kp2, good_matches):
    """
    Find homography, draw detected object, and visualize matches.
    """
    if len(good_matches) > MIN_MATCH_COUNT_HIGH:
        # Extract points from good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Find homography
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Draw detected object's bounding box
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

        print(f"Object detected. Matches: {len(good_matches)}")
    else:
        print(
            f"Not enough matches are found: {len(good_matches)}/{MIN_MATCH_COUNT_HIGH}"
        )
        matches_mask = None

    # Draw matches
    draw_params = dict(
        matchColor=(0, 255, 0),  # Matches in green
        singlePointColor=None,
        matchesMask=matches_mask,  # Draw only inliers
        flags=2,
    )
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    return img3


def process_scene(
    object_descs, scene_left, scene_front, scene_right, object_names, scene_name
):
    matches = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    # SCENE LEFT
    # Detect and compute scene features
    kps, scene_desc = detect_and_compute(scene_left, sift)
    
    for i in range(len(object_descs)):
        # Match features and apply Lowe's ratio test
        good_matches = match_features(object_descs[i], scene_desc)
        
        matches[i][0] = len(good_matches)

    # SCENE FRONT
    # Detect and compute scene features
    _, scene_desc = detect_and_compute(scene_front, sift)
    for i in range(len(object_descs)):
        # Match features and apply Lowe's ratio test
        good_matches = match_features(object_descs[i], scene_desc)
        matches[i][1] = len(good_matches)

    # SCENE RIGHT
    # Detect and compute scene features
    _, scene_desc = detect_and_compute(scene_right, sift)
    for i in range(len(object_descs)):
        # Match features and apply Lowe's ratio test
        good_matches = match_features(object_descs[i], scene_desc)
        matches[i][2] = len(good_matches)

    detected_objects = []
    i = 0
    for object_results in matches:
        if (
            max(object_results) > MIN_MATCH_COUNT_HIGH
            or sum(object_results) >= MIN_MATCH_COUNT_SUM
        ):
            detected_objects.append(object_names[i])
        i += 1

    print("Objects detected in ", scene_name, ": ", detected_objects)
    print(
        matches,
        sum(matches[0]),
        sum(matches[1]),
        sum(matches[2]),
        sum(matches[3]),
        sum(matches[4]),
        sum(matches[5]),
        sum(matches[6]),
        sum(matches[7]),
        sum(matches[8]),
        sum(matches[9]),
    )

    return detected_objects


def test_detection(img1, img2):

    # Detect and compute features
    kp1, des1 = detect_and_compute(img1, sift)
    kp2, des2 = detect_and_compute(img2, sift)

    # Match features and apply Lowe's ratio test
    good_matches = match_features(des1, des2)

    print("# of matches: ", good_matches)
    # Find homography and draw results
    result_img = find_homography_and_draw(img1, img2, kp1, kp2, good_matches)

    # Display results
    plt.imshow(result_img, "gray")
    plt.title("Feature Matching with SIFT")
    plt.show()


def write_results(results, output_file):
    """Write detection results to a text file."""
    with open(output_file, "w") as f:
        for scene_name, detected_objects in results:
            detected_str = ", ".join(detected_objects) if detected_objects else "None"
            f.write(f"{scene_name}: {detected_str}\n")


# Main Script
if __name__ == "__main__":
    # Paths to datasets
    objects_folder = "Objects"
    scenes_folder = "Scenes"
    output_file = "testing.txt"

    # Load images
    object_images, object_names = load_images(objects_folder)
    print("Object images loaded")
    scene_images, scene_names = load_images(scenes_folder)
    print("Scene images loaded")

    # Initialize SIFT
    sift = cv.SIFT_create()

    # Find all object descs
    object_descs = []
    for object in object_images:
        _, desc = detect_and_compute(object, sift)
        object_descs.append(desc)

    # Process scenes
    results = []
    for i in range(0, len(scene_images), 3):
        scene_front = scene_images[i]
        scene_left = scene_images[i + 1]
        scene_right = scene_images[i + 2]
        scene_name = scene_names[i]
        objects = process_scene(
            object_descs, scene_left, scene_front, scene_right, object_names, scene_name
        )
        results.append((scene_name, objects))

    write_results(results, output_file)

    print(f"Task 1 completed using SIFT. Results written to {output_file}.")
