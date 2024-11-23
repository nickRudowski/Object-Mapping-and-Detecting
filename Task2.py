import cv2
import numpy as np


def resize_object(obj, max_height):
    h, w = obj.shape[:2]
    
    if h > max_height:

        scale_factor = max_height / h
        obj = cv2.resize(obj, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    
    return obj


def extract_non_transparent_keypoints(obj):
    bgr, alpha = obj[:, :, :3], obj[:, :, 3]
    mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]  
    obj_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    obj_gray_masked = cv2.bitwise_and(obj_gray, obj_gray, mask=mask)  

    return obj_gray_masked, mask


def compute_tight_bounding_box(points):
    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
    return (int(x_min), int(y_min)), (int(x_max), int(y_max))


def detect_objects(object_images, scene_image_path, output_path, colors, obj_labels):
    
    scene = cv2.imread(scene_image_path)
    if scene is None:
        print("Error: no scene image.")
        return

    
    scene_height, scene_width = scene.shape[:2]
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    
    output_image = scene.copy()

    
    sift = cv2.SIFT_create()

    
    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_gray, None)

    
    for obj_idx, obj_path in enumerate(object_images):
        
        obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
        if obj is None:
            print(f"Error: no object image: {obj_path}")
            continue

        
        obj = resize_object(obj, scene_height)

        
        obj_gray, mask = extract_non_transparent_keypoints(obj)

        
        obj_keypoints, obj_descriptors = sift.detectAndCompute(obj_gray, mask)

        
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(obj_descriptors, scene_descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) >= 10:  
            
            obj_pts = np.float32([obj_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            scene_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            
            M, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 5.0)

            if M is not None:
                
                inliers = mask.ravel().astype(bool)
                scene_inlier_pts = scene_pts[inliers].reshape(-1, 2)

                
                top_left, bottom_right = compute_tight_bounding_box(scene_inlier_pts)

                
                cv2.rectangle(output_image, top_left, bottom_right, colors[obj_idx], 4)

                
                label = f"{obj_labels[obj_idx]}"
                cv2.putText(output_image, label, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, colors[obj_idx], 6)
            else:
                print(f"Failed to ccalculaate homography for {obj_path}.")
        else:
            res = cv2.matchTemplate(scene_gray, obj_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.5:  
                top_left = max_loc
                h, w = obj_gray.shape
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(output_image, top_left, bottom_right, colors[obj_idx], 5)
                label = f"{obj_labels[obj_idx]}"
                cv2.putText(output_image, label, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, colors[obj_idx], 6)

    cv2.imwrite(output_path, output_image)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    object_images = [
        "Objects/O1.png",
        "Objects/O2.png",
        "Objects/O3.png",
        "Objects/O4.png",
        "Objects/O5.png",
        "Objects/O6.png",
        "Objects/O7.png",
        "Objects/O8.png",
        "Objects/O9.png",
        "Objects/O10.png"
    ]
    object_labels = [
        "Water Dispenser",
        "Kettle",
        "Ziploc",
        "Tea pot",
        "Lunch box",
        "Flour",
        "Pan",
        "Box",
        "Tissues",
        "Hat"
    ]

    scene_image_path = "output_panorama.jpg"

    output_path = "Detected/panorama_detect.png"

    colors = (
        (0, 0, 255),
        (0, 127, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (127, 0, 255),
        (255, 255, 0),
        (255, 127, 0),
        (50, 205, 50)
    )


    detect_objects(object_images, scene_image_path, output_path, colors, object_labels)

    # mass produce    
    for idx, obj in enumerate(object_images):
         print(f"Detecting for scene {idx + 1}")
         scene_image_path = f"Scenes/S{idx + 1}_front.jpg"
         output_path = f"Detected/front_{idx + 1}.png"
         detect_objects(object_images[:(idx + 1)], scene_image_path, output_path, colors, object_labels)

         print(f"Detecting for scene {idx + 1}")
         scene_image_path = f"Scenes/S{idx + 1}_left.jpg"
         output_path = f"Detected/left_{idx + 1}.png"
         detect_objects(object_images[:(idx + 1)], scene_image_path, output_path, colors, object_labels)

         print(f"Detecting for scene {idx + 1}")
         scene_image_path = f"Scenes/S{idx + 1}_right.jpg"
         output_path = f"Detected/right_{idx + 1}.png"
         detect_objects(object_images[:(idx + 1)], scene_image_path, output_path, colors, object_labels)