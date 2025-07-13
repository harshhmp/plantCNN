import darknet
import cv2

def main():
    configFile = "cfg/yolov4.cfg"
    dataFile = "cfg/coco.data"
    weights = "yolov4.weights"

    network, class_names, class_colors = darknet.load_network(configFile, dataFile, weights, batch_size=1)

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    
    images = []
    run = 1
    
    while(run):
        user_input = input("Image name (dog.jpg) exit to stop:")
        
        if user_input == "exit":
            run = 0
        else:
            image_path_start = "data/"
            image = cv2.imread(image_path_start + user_input)
            images.append(image)
    
    runMultipleDetections(images, network, class_names, class_colors, width, height)
    
def runMultipleDetections(images, network, class_names, class_colors, width, height):
    file_parth_start = "output/"
    file_path_end = ".jpg"
    image_num = 0
    for image in images:
        complete_output_path = file_parth_start + str(image_num) + file_path_end
        image_num += 1
        
        image_resized = cv2.resize(image, (width, height))

        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)

        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            left, top = int(x - w / 2), int(y - h / 2)
            right, bottom = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(image_resized, (left, top), (right, bottom), class_colors[label], 2)
            cv2.putText(image_resized, f"{label} [{confidence}]", (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[label], 2)
        
        cv2.imwrite(complete_output_path, image_resized)

def runSingleDetection(image, network, class_names, class_colors, width, height):
    image_resized = cv2.resize(image, (width, height))

    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)

    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        left, top = int(x - w / 2), int(y - h / 2)
        right, bottom = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(image_resized, (left, top), (right, bottom), class_colors[label], 2)
        cv2.putText(image_resized, f"{label} [{confidence}]", (left, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[label], 2)
        
    cv2.imwrite("output.jpg", image_resized)

    cv2.imshow("Detection", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()