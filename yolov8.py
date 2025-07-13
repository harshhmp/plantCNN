from inference import get_model
import cv2
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr

image = load_image_bgr("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.rd.com%2Fwp-content%2Fuploads%2F2022%2F11%2FRD-flowering-house-plants-GettyImages-1361899895-JVedit.jpg%3Fresize%3D2048&f=1&nofb=1&ipt=b1918ccc89b105609bd6372da3551bc2553abe17e7c1f60f50dc599e52cb7709")

model = get_model(model_id="yolov8n-640")

results = model.infer(image)[0]
results = sv.Detections.from_inference(results)

annotator = sv.BoxAnnotator(thickness=4)
annotated_image = annotator.annotate(image, results)
annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)

annotated_image = annotator.annotate(annotated_image, results)

cv2.imwrite("output/out.jpg", annotated_image)