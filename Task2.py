import cv2
import numpy as np

def detect_and_draw_bounding_boxes(object_images, scene_image_path, output_path):
    scene_image = cv2.imread(scene_image_path)
    scene_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    scene_gray = clahe.apply(scene_gray)
    scene_gray = cv2.GaussianBlur(scene_gray, (5, 5), 0)

    sift = cv2.SIFT_create()

    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_gray, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for idx, obj_image_path in enumerate(object_images):
        obj_image = cv2.imread(obj_image_path)
        obj_gray = cv2.cvtColor(obj_image, cv2.COLOR_BGR2GRAY)
        obj_gray = cv2.GaussianBlur(obj_gray, (5, 5), 0)


        obj_keypoints, obj_descriptors = sift.detectAndCompute(obj_gray, None)
        matches = flann.knnMatch(obj_descriptors, scene_descriptors, k=2)

        

        good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

        if len(good_matches) > 10:
            obj_pts = np.float32([obj_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            scene_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 2.0)

            if matrix is not None:
                h, w = obj_gray.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                projected_corners = cv2.perspectiveTransform(corners, matrix)

                label = f"Object {idx + 1}"
                cv2.polylines(scene_image, [np.int32(projected_corners)], True, (0, 255, 0), 3)
                x, y = np.int32(projected_corners[0][0])
                cv2.putText(scene_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)

    cv2.imwrite(output_path, scene_image)
    scene_image = cv2.resize(scene_image, (800, 600))
    cv2.imshow("Detected Objects", scene_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":

    # Paths to object images
    object_images = ["Dataset/Objects/O1.jpg", "Dataset/Objects/O2.jpg", "Dataset/Objects/O3.jpg", "Dataset/Objects/O4.jpg", "Dataset/Objects/O5.jpg"]

    # Path to the scene image
    scene_image_path = "Dataset/Scenes/S5_front.jpg"

    # Path to save the output image
    output_path = "output_with_boxes.jpg"

    # Run the detection and drawing function
    detect_and_draw_bounding_boxes(object_images, scene_image_path, output_path)