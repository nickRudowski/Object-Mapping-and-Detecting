import cv2
from google.colab.patches import cv2_imshow
import os


object_images_folder = "/content/Objects" 
scene_images_folder = "/content/Scene"   

# Function to detect objects in a single scene
def detect_objects_in_scene(scene_image_path, object_images):
    scene_image = cv2.imread(scene_image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=100)  #keypoints
    detected_objects = []

    for obj_id, obj_path in object_images.items():
        object_image = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)

        keypoints_obj, descriptors_obj = sift.detectAndCompute(object_image, None)
        keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_image, None)

        # Match descriptors
        index_params = dict(algorithm=1, trees=5)  
        search_params = dict(checks=50)  
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_obj, descriptors_scene, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:  
            if m.distance < 0.75 * n.distance:  
                good_matches.append(m)
        match_threshold = max(10, len(keypoints_obj) * 0.05)
        if len(good_matches) >= match_threshold:  
            detected_objects.append(obj_id)

    return detected_objects

object_images = {
    obj.split('.')[0]: os.path.join(object_images_folder, obj)
    for obj in os.listdir(object_images_folder)
    if obj.endswith((".jpg", ".png"))
}

scene_images = [
    os.path.join(scene_images_folder, scene)
    for scene in os.listdir(scene_images_folder)
    if scene.endswith((".jpg", ".png"))
]

results = {}
for scene_index, scene_image_path in enumerate(scene_images, start=1):
    detected_objects = detect_objects_in_scene(scene_image_path, object_images)
    results[f"Scene {scene_image_path}"] = detected_objects

for scene, detected_objects in results.items():
    print(f"{scene}: {', '.join(detected_objects)}")
