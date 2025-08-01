import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

modelB0 = EfficientNetB0(weights='imagenet')
modelCustom = tf.keras.models.load_model("../efficientnet_custom_model_1.keras")

def getModel1ClassNames():
    
    class_names = ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Bacterial leaf blight in rice leaf', 'Blight in corn Leaf', 'Blueberry healthy', 'Brown spot in rice leaf', 'Cercospora leaf spot', 'Cherry (including sour) Powdery mildew', 'Cherry (including_sour) healthy', 'Common Rust in corn Leaf', 'Corn (maize) healthy', 'Garlic', 'Grape Black rot', 'Grape Esca Black Measles', 'Grape Leaf blight Isariopsis Leaf Spot', 'Grape healthy', 'Gray Leaf Spot in corn Leaf', 'Leaf smut in rice leaf', 'Nitrogen deficiency in plant', 'Orange Haunglongbing Citrus greening', 'Peach healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Sogatella rice', 'Soybean healthy', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot', 'Tomato Tomato mosaic virus', 'Tomato healthy', 'Waterlogging in plant', 'algal leaf in tea', 'anthracnose in tea', 'bird eye spot in tea', 'brown blight in tea', 'cabbage looper', 'corn crop', 'ginger', 'healthy tea leaf', 'lemon canker', 'onion', 'potassium deficiency in plant', 'potato crop', 'potato hollow heart', 'red leaf spot in tea', 'tomato canker']
    
    return class_names

def getModel2ClassNames():
    
    class_names = ['Apple__black_rot', 'Apple__healthy', 'Apple__rust', 'Apple__scab', 'Cassava__bacterial_blight', 'Cassava__brown_streak_disease', 'Cassava__green_mottle', 'Cassava__healthy', 'Cassava__mosaic_disease', 'Cherry__healthy', 'Cherry__powdery_mildew', 'Chili__healthy', 'Chili__leaf curl', 'Chili__leaf spot', 'Chili__whitefly', 'Chili__yellowish', 'Coffee__cercospora_leaf_spot', 'Coffee__healthy', 'Coffee__red_spider_mite', 'Coffee__rust', 'Corn__common_rust', 'Corn__gray_leaf_spot', 'Corn__healthy', 'Corn__northern_leaf_blight', 'Cucumber__diseased', 'Cucumber__healthy', 'Gauva__diseased', 'Gauva__healthy', 'Grape__black_measles', 'Grape__black_rot', 'Grape__healthy', 'Grape__leaf_blight_(isariopsis_leaf_spot)', 'Jamun__diseased', 'Jamun__healthy', 'Lemon__diseased', 'Lemon__healthy', 'Mango__diseased', 'Mango__healthy', 'Peach__bacterial_spot', 'Peach__healthy', 'Pepper_bell__bacterial_spot', 'Pepper_bell__healthy', 'Pomegranate__diseased', 'Pomegranate__healthy', 'Potato__early_blight', 'Potato__healthy', 'Potato__late_blight', 'Rice__brown_spot', 'Rice__healthy', 'Rice__hispa', 'Rice__leaf_blast', 'Rice__neck_blast', 'Soybean__bacterial_blight', 'Soybean__caterpillar', 'Soybean__diabrotica_speciosa', 'Soybean__downy_mildew', 'Soybean__healthy', 'Soybean__mosaic_virus', 'Soybean__powdery_mildew', 'Soybean__rust', 'Soybean__southern_blight', 'Strawberry___leaf_scorch', 'Strawberry__healthy', 'Sugarcane__bacterial_blight', 'Sugarcane__healthy', 'Sugarcane__red_rot', 'Sugarcane__red_stripe', 'Sugarcane__rust', 'Tea__algal_leaf', 'Tea__anthracnose', 'Tea__bird_eye_spot', 'Tea__brown_blight', 'Tea__healthy', 'Tea__red_leaf_spot', 'Tomato__bacterial_spot', 'Tomato__early_blight', 'Tomato__healthy', 'Tomato__late_blight', 'Tomato__leaf_mold', 'Tomato__mosaic_virus', 'Tomato__septoria_leaf_spot', 'Tomato__spider_mites_(two_spotted_spider_mite)', 'Tomato__target_spot', 'Tomato__yellow_leaf_curl_virus', 'Wheat__brown_rust', 'Wheat__healthy', 'Wheat__septoria', 'Wheat__yellow_rust']

    return class_names

def runEfficientNet(image_array):
    image_array = preprocess_input(image_array)
    
    predictions = modelB0.predict(image_array)

    return decode_predictions(predictions, top=1)[0]
    
def runCustomModel(image_array):
    # Predict image and translate result into words
    prediction = modelCustom.predict(image_array)
    confidence = float(prediction.max()) * 100
    
    class_index = prediction.argmax()
    
    CLASS_NAMES = getModel1ClassNames()
    result = CLASS_NAMES[class_index]

    return confidence, result