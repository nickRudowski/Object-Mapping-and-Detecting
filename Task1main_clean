# Drier Code for Task 1
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Constants for feature matching and image processing
MIN_MATCH_COUNT_HIGH = 50  # Minimum matches to detect an object
MIN_MATCH_COUNT_SUM = 140  # Minimum total matches across views
RATIO_TEST_THRESHOLD = 0.70  # Lowe's ratio test threshold
MEDIAN_KERNEL = 1  # Median blur kernel size


def resize_image(image, max_width=2048, max_height=2048):
    """
    Resize the image to fit within max dimensions while preserving the aspect ratio.
    Parameters:
        image: Input image.
        max_width: Maximum allowed width.
        max_height: Maximum allowed height.
    Returns:
        Resized image.
    """
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv.resize(image, new_size, interpolation=cv.INTER_AREA)
    return image


def load_images(folder_path):
    """
    Load all images from a given folder.
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
            image = resize_image(cv.imread(img_path, cv.IMREAD_GRAYSCALE))
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Normalize and apply median blur
            cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)
            image = cv.medianBlur(image, MEDIAN_KERNEL)

            images.append(image)
            image_names.append(filename)

    return images, image_names


def detect_and_compute(image, sift):
    """
    Detect keypoints and compute descriptors using SIFT.
    Parameters:
        image: Input image.
        sift: SIFT object.
    Returns:
        Keypoints and descriptors.
    """
    return sift.detectAndCompute(image, None)


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


def find_homography_and_draw(img1, img2, kp1, kp2, good_matches):
    """
    Find homography, draw detected object, and visualize matches.
    Parameters:
        img1: Object image.
        img2: Scene image.
        kp1: Keypoints in object image.
        kp2: Keypoints in scene image.
        good_matches: List of good matches.
    Returns:
        Image with drawn matches and bounding box (if homography is found).
    """
    if len(good_matches) > MIN_MATCH_COUNT_HIGH:
        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Draw bounding box on the scene image
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        print(f"Object detected. Matches: {len(good_matches)}")
    else:
        print(f"Not enough matches: {len(good_matches)}/{MIN_MATCH_COUNT_HIGH}")
        matches_mask = None

    # Draw matches
    draw_params = dict(
        matchColor=(0, 255, 0),  # Matches in green
        singlePointColor=None,
        matchesMask=matches_mask,  # Draw only inliers
        flags=2,
    )
    return cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)


def process_scene(object_descs, scene_left, scene_front, scene_right, object_names, scene_name):
    """
    Process a scene to detect objects by matching features across views.
    Parameters:
        object_descs: Descriptors of objects.
        scene_left: Left view of the scene.
        scene_front: Front view of the scene.
        scene_right: Right view of the scene.
        object_names: Names of objects.
        scene_name: Name of the scene.
    Returns:
        List of detected objects.
    """
    matches = np.zeros((len(object_descs), 3), dtype=int)  # Match counts for each object across views

    # Process left view
    _, scene_desc = detect_and_compute(scene_left, sift)
    matches[:, 0] = [len(match_features(desc, scene_desc)) for desc in object_descs]

    # Process front view
    _, scene_desc = detect_and_compute(scene_front, sift)
    matches[:, 1] = [len(match_features(desc, scene_desc)) for desc in object_descs]

    # Process right view
    _, scene_desc = detect_and_compute(scene_right, sift)
    matches[:, 2] = [len(match_features(desc, scene_desc)) for desc in object_descs]

    # Identify detected objects
    detected_objects = [
        object_names[i]
        for i in range(len(matches))
        if max(matches[i]) > MIN_MATCH_COUNT_HIGH or sum(matches[i]) >= MIN_MATCH_COUNT_SUM
    ]
    print(f"Objects detected in {scene_name}: {detected_objects}")
    return detected_objects


def write_results(results, output_file):
    """
    Write detection results to a text file.
    Parameters:
        results: List of scene results.
        output_file: Path to the output file.
    """
    with open(output_file, "w") as f:
        for scene_name, detected_objects in results:
            detected_str = ", ".join(detected_objects) if detected_objects else "None"
            f.write(f"{scene_name}: {detected_str}\n")


# Main Script
if __name__ == "__main__":
    # Paths to datasets, replace with file locations
    objects_folder = r"C:\Users\Dylan\Documents\CP467 Image\467 Proj\Objects"
    scenes_folder = r"C:\Users\Dylan\Documents\CP467 Image\467 Proj\Scenes"
    output_file = r"C:\Users\Dylan\Documents\CP467 Image\467 Proj\testing.txt"

    # Load object and scene images
    object_images, object_names = load_images(objects_folder)
    scene_images, scene_names = load_images(scenes_folder)

    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Compute descriptors for all object images
    object_descs = [detect_and_compute(img, sift)[1] for img in object_images]

    # Process scenes and detect objects
    results = []
    for i in range(0, len(scene_images), 3):
        scene_name = scene_names[i]
        detected_objects = process_scene(
            object_descs,
            scene_images[i + 1],  # Left view
            scene_images[i],      # Front view
            scene_images[i + 2],  # Right view
            object_names,
            scene_name
        )
        results.append((scene_name, detected_objects))

    # Write results to file
    write_results(results, output_file)
    print(f"Task completed using SIFT. Results written to {output_file}.")
