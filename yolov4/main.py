import darknet
import cv2

configFile = "cfg/yolov4.cfg"
dataFile = "cfg/coco.data"
weights = "yolov4.weights"

network, class_names, class_colors = darknet.load_network(configFile, dataFile, weights, batch_size=1)

width = darknet.network_width(network)
height = darknet.network_height(network)

image_path = "data/human.jpg"
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (width, height))

darknet_image = darknet.make_image(width, height, 3)
darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)

for label, confidence, bbox in detections:
    x, y, w, h = bbox
    left, top = int(x - w / 2), int(y - h / 2)
    right, bottom = int(x + w / 2), int(y + h / 2)
    cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image_resized, f"{label} [{confidence}]", (left, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
cv2.imwrite("output.jpg", image_resized)

cv2.imshow("Detection", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()