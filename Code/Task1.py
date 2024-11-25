# Drier Code for Task 1
import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Constants for feature matching and image processing
MIN_MATCH_COUNT_HIGH = 140  # Minimum matches to detect an object
MIN_MATCH_COUNT_SUM = 120  # Minimum total matches across views
RATIO_TEST_THRESHOLD = 0.70  # Lowe's ratio test threshold
INLIER_RATIO_THRESHOLD = 0.6  # Threshold for homography validiy ratio
RANSAC_REPROJECTION_THRESHOLD = 5  # Threshold for RANSAC homography inlier detection

OBJECTS = {
    "O1": ["Soda Machine", [], [], []],
    "O2": ["Moka Pot", [], [], []],
    "O3": ["Ziploc Box", [], [], []],
    "O4": ["Teapot", [], [], []],
    "O5": ["Lunchbox", [], [], []],
    "O6": ["Flour", [], [], []],
    "O7": ["Frying pan", [], [], []],
    "O8": ["Box", [], [], []],
    "O9": ["Tissue Box", [], [], []],
    "O10": ["Hat", [], [], []],
}


def resize_image(obj, max_height):
    h, w = obj.shape[:2]

    if h > max_height:

        scale_factor = max_height / h
        obj = cv.resize(
            obj,
            (int(w * scale_factor), int(h * scale_factor)),
            interpolation=cv.INTER_AREA,
        )

    return obj


def extract_non_transparent_keypoints(obj):
    bgr, alpha = obj[:, :, :3], obj[:, :, 3]
    mask = cv.threshold(alpha, 1, 255, cv.THRESH_BINARY)[1]
    obj_gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    obj_gray_masked = cv.bitwise_and(obj_gray, obj_gray, mask=mask)

    return obj_gray_masked, mask


def load_scene_images(folder_path):
    """
    Load all scene images from a given folder.
    Parameters:
        folder_path: Path to the folder containing images.
    Returns:
        A list of loaded images and their filenames.
    """
    images = []
    image_names = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Load images in sorted order
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            print(f"Loading image: {img_path}")

            # Read and process the image
            image = resize_image(cv.imread(img_path, cv.IMREAD_GRAYSCALE), 2048)

            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Normalize 
            cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)

            images.append(image)
            image_names.append(filename[:-4])

    return images, image_names


def load_obj_images(folder_path, sift):
    """
    Load all object images from a given folder to the OBJECTS dictionary.
    Parameters:
        folder_path: Path to the folder containing images.
    Returns:
        A list of loaded images and their filenames.
    """

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Load images in sorted order
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            print(f"Loading image: {img_path}")

            # Read and process the image
            image = resize_image(cv.imread(img_path, cv.IMREAD_UNCHANGED), 2048)

            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Normalize 
            cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)

            name = filename[:-4]
            # Compute descriptors only in non-transparent part of object for all objects
            obj_gray, mask = extract_non_transparent_keypoints(image)
            obj_keypoints, obj_descriptors = sift.detectAndCompute(obj_gray, mask)
            OBJECTS[name][1] = obj_keypoints
            OBJECTS[name][2] = obj_descriptors
            OBJECTS[name][3] = image


def match_features(des1, des2, ratio=RATIO_TEST_THRESHOLD):
    """
    Match features between two sets of descriptors using FLANN and apply Lowe's ratio test.
    Parameters:
        des1: Descriptors from image 1.
        des2: Descriptors from image 2.
        ratio: Lowe's ratio test threshold.
    Returns:
        List of good matches.
    """
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Find matches and apply ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches


def localize_homography(good_matches, object_image, obj_kps, scene_kps):
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        obj[i, 0] = obj_kps[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = obj_kps[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = scene_kps[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = scene_kps[good_matches[i].trainIdx].pt[1]
    H, mask = cv.findHomography(obj, scene, cv.RANSAC, RANSAC_REPROJECTION_THRESHOLD)
    h = object_image.shape[0]
    w = object_image.shape[1]

    obj_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        -1, 1, 2
    )
    scene_corners = cv.perspectiveTransform(obj_corners, H)
    return mask, scene_corners, H


def decompose_homography(lis):
    """Decompose homography matrix into rotation, scale, and shear components."""
    try:
        # Homography matrix should be 3x3
        H = np.array(lis)
        assert H.shape == (3, 3), "Homography matrix should be 3x3."

        # Normalize H to make sure it's in a consistent scale
        H_normalized = H / H[2, 2]

        # Extract rotation and scale using SVD
        U, S, Vt = np.linalg.svd(H_normalized[:2, :2])
        rotation = np.dot(U, Vt)
        scale = np.diag(S)

        # Extract translation vector
        translation = H_normalized[:2, 2]

        return rotation, scale, translation

    except np.linalg.LinAlgError:
        # print("Error: SVD did not converge.")
        return [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]


def process_scene(
    scene_left,
    scene_front,
    scene_right,
    scene_file,
    sift,
):
    """
    Process a scene to detect objects by matching features across views.
    Parameters:
        scene_left: Left view of the scene.
        scene_front: Front view of the scene.
        scene_right: Right view of the scene.
        object_names: Names of objects.
        scene_name: Name of the scene.
    Returns:
        List of detected objects.
    """

    scene_name = "Scene " + scene_file[1:3].replace("_", "")
    print(f"Analyzing {scene_name}")

    # Process left, front, and right views
    scene_kps_left, scene_desc_left = sift.detectAndCompute(scene_left, None)
    scene_kps_front, scene_desc_front = sift.detectAndCompute(scene_front, None)
    scene_kps_right, scene_desc_right = sift.detectAndCompute(scene_right, None)

    detected_objects = []

    for obj_ID in OBJECTS:
        obj_data = OBJECTS[obj_ID]
        obj_kps = obj_data[1]
        obj_desc = obj_data[2]
        obj_image = obj_data[3]
        name = obj_data[0] + " (" + obj_ID + ")"

        # Compute matches
        good_matches_left = match_features(obj_desc, scene_desc_left)
        good_matches_front = match_features(obj_desc, scene_desc_front)
        good_matches_right = match_features(obj_desc, scene_desc_right)

        # Compute homography masks for thresholding and object localization
        if len(good_matches_left) > 30:
            mask_left, _, Hleft = localize_homography(
                good_matches_left, obj_image, obj_kps, scene_kps_left
            )
        else:
            mask_left, _ = ([0], np.empty((4, 1, 2), dtype=np.float32))
            Hleft = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        if len(good_matches_front) > 30:
            mask_front, _, Hfront = localize_homography(
                good_matches_front, obj_image, obj_kps, scene_kps_front
            )
        else:
            mask_front, _ = (
                [0],
                np.empty((4, 1, 2), dtype=np.float32),
            )
            Hfront = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        if len(good_matches_right) > 30:
            mask_right, _, Hright = localize_homography(
                good_matches_right, obj_image, obj_kps, scene_kps_right
            )
        else:
            mask_right, _ = (
                [0],
                np.empty((4, 1, 2), dtype=np.float32),
            )
            Hright = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # Compute inlier ratios based on homography matrix
        inlier_ratio_left = np.sum(mask_left) / len(mask_left)
        inlier_ratio_front = np.sum(mask_front) / len(mask_front)
        inlier_ratio_right = np.sum(mask_right) / len(mask_right)

        # Compute summary values
        max_matches = max(
            len(good_matches_left), len(good_matches_front), len(good_matches_right)
        )
        sum_matches = (
            len(good_matches_left) + len(good_matches_front) + len(good_matches_right)
        )
        max_inlier_ratio = max(
            inlier_ratio_left, inlier_ratio_front, inlier_ratio_right
        )
        # Compute avg of distances among top 50 matches
        sorted_matches = sorted(
            (good_matches_left + good_matches_front + good_matches_right),
            key=lambda match: match.distance,
        )[:10]

        rotation_left = np.zeros((2, 2))
        rotation_front = np.zeros((2, 2))
        rotation_right = np.zeros((2, 2))

        # Check homography matrices and decompose them if non-zero
        if np.all(Hleft != 0):
            rotation_left, _, _ = decompose_homography(Hleft)
        if np.all(Hfront != 0):
            rotation_front, _, _ = decompose_homography(Hfront)
        if np.all(Hright != 0):
            rotation_right, _, _ = decompose_homography(Hright)

        # Calculate rotation consistency
        rotation_consistency = (
            np.arctan2(rotation_left[1, 0], rotation_left[0, 0]),
            np.arctan2(rotation_front[1, 0], rotation_front[0, 0]),
            np.arctan2(rotation_right[1, 0], rotation_right[0, 0]),
        )
        rotation_validity = 10

        if rotation_consistency != (np.float64(0.0), np.float64(0.0), np.float64(0.0)):
            pairwise_differences = np.abs(
                np.array(rotation_consistency)[:, None] - np.array(rotation_consistency)
            )
            # Extract the smallest difference, ignoring zeros (difference with itself)
            rotation_validity = np.min(
                pairwise_differences[np.nonzero(pairwise_differences)]
            )

        def calculate_score(rotation_val, max_matches, sum_matches, max_inlier_ratio):
            score = 0
            if rotation_val < 0.5:
                score += 1
            if max_matches > 100:
                score += 1
            if sum_matches > 113:
                score += 1
            if max_inlier_ratio > 0.45:
                score += 1
            return score

        score = calculate_score(
            round(rotation_validity, 2),
            max_matches,
            sum_matches,
            round(max_inlier_ratio, 2),
        )

        if score >= 2:
            detected_objects.append(name)

    results = (
        f"Objects detected in {scene_file[:3]}left.jpg: {detected_objects}\n"
        + f"Objects detected in {scene_file}.jpg: {detected_objects}\n"
        + f"Objects detected in {scene_file[:3]}right.jpg: {detected_objects}\n"
    )
    return results


def write_results(results, output_file):
    """
    Write detection results to a text file.
    Parameters:
        results: List of scene results.
        output_file: Path to the output file.
    """
    with open(output_file, "w") as f:
        for i, detected_objects in enumerate(results):
            if i == 0:
                temp = str(detected_objects) + "\n"
            else:
                f.write(str(detected_objects) + "\n")
        f.write(temp)


# Main Script
if __name__ == "__main__":
    # Paths to datasets, replace with file locations
    objects_folder = "Objects"
    scenes_folder = "Scenes"
    output_file = "testing.txt"

    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Load object images and populate object dictionary with descriptors, keypoints and images
    load_obj_images(objects_folder, sift)

    # Load object and scene images
    scene_images, scene_names = load_scene_images(scenes_folder)
    
    # Process scenes and detect objects
    results = []
    for i in range(0, len(scene_images), 3):
        detected_objects = process_scene(
            scene_images[i + 1],  # Left view
            scene_images[i],  # Front view
            scene_images[i + 2],  # Right view
            scene_names[i],
            sift,
        )
        results.append(detected_objects)

    # Write results to file
    write_results(results, output_file)
    print(f"Task 1 completed using SIFT. Results written to {output_file}.")
