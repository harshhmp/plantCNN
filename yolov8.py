from inference import get_model
import cv2
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    trainCNNModel()
    
def runYolo(): 
    image = load_image_bgr("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.rd.com%2Fwp-content%2Fuploads%2F2022%2F11%2FRD-flowering-house-plants-GettyImages-1361899895-JVedit.jpg%3Fresize%3D2048&f=1&nofb=1&ipt=b1918ccc89b105609bd6372da3551bc2553abe17e7c1f60f50dc599e52cb7709")

    model = get_model(model_id="yolov8n-640")

    results = model.infer(image)[0]
    results = sv.Detections.from_inference(results)

    annotator = sv.BoxAnnotator(thickness=4)
    annotated_image = annotator.annotate(image, results)
    annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)

    annotated_image = annotator.annotate(annotated_image, results)

    cv2.imwrite("output/out.jpg", annotated_image)
    cv2.imwrite("")

def trainCNNModel():
    x_train = tf.keras.utils.image_dataset_from_directory(
        "plantData/Image Data base/Image Data base",
        validation_split = 0.2, 
        subset="training",
        seed=123,
        image_size=(128, 128),
        batch_size=64
    )
    
    x_test = tf.keras.utils.image_dataset_from_directory(
        "plantData/Image Data base/Image Data base",
        validation_split = 0.2,
        subset="validation",
        seed=123,
        image_size=(128,128),
        batch_size=64
    )
    
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(128, 128, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(x_train.class_names), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(x_train, validation_data=x_test, epochs=5)
    
    model.save("model1")

if __name__ == "__main__":
    main()