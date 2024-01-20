from Detector import *
from Translate import *

import streamlit as st
import cv2

# modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'

classFile = 'coco.names'
imagePath = "/test/000000000632_jpg.rf.bfddd0d2b86ac2bf1e80624422433f0a.jpg"
threshold = 0.5
API_KEY = '471b0e0033ab103162ac'
selected_language = None

detector = Detector()
translate = Translate()

detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()


# detector.predictImage(imagePath, threshold)
def capture():
    pass


def main():
    st.title("Object Detection with TensorFlow")

    language_options = ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque', 'Belarusian',
                        'Bengali', 'Bosnian', 'Bulgarian', 'Catalan', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English',
                        'Esperanto', 'Estonian', 'Filipino', 'Finnish', 'French', 'Galician', 'Georgian', 'German',
                        'Greek',
                        'Gujarati', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Irish', 'Italian',
                        'Japanese',
                        'Javanese', 'Kannada', 'Kazakh', 'Korean', 'Latin', 'Latvian', 'Lithuanian', 'Macedonian',
                        'Malay',
                        'Malayalam', 'Maltese', 'Marathi', 'Mongolian', 'Nepali', 'Norwegian', 'Persian', 'Polish',
                        'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Sinhalese', 'Slovak', 'Slovenian',
                        'Spanish', 'Swahili', 'Swedish', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu',
                        'Uzbek',
                        'Vietnamese', 'Welsh', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']

    col1, col2 = st.columns(2)
    with col1:
        st.text("Choose your language : ")
    with col2:
        selected_language = col2.selectbox('', language_options)
        translate.target_language = selected_language

    # # cap = cv2.VideoCapture(0)
    # # capture_button = st.button("Capture")
    # image = st.camera_input("Take a picture")
    # # if not cap.isOpened():
    # #     st.error("Error: Unable to open camera.")
    # #     return
    #
    # # if capture_button:
    # if image:
    #     # ret, frame = cap.read()
    #
    #     # making predictions on captured image
    #     # image_with_bounding_box = detector.createBoundingBox(frame, threshold)
    #     bytes_data = image.getvalue()
    #     image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    #
    #     image_with_bounding_box = detector.createBoundingBox(image, threshold)
    #
    #     #displaying images
    #     org, boxed = st.columns(2)
    #     with org:
    #         # st.image(frame, channels="BGR", use_column_width=True)
    #         st.image(image, channels="BGR", use_column_width=True)
    #     with boxed:
    #         st.image(image_with_bounding_box, channels="BGR", use_column_width=True)

    image = st.camera_input("Capture")
    if image:
        bytes_data = image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        image_with_bounding_boxes = detector.createBoundingBox(image, threshold)
        st.image(image_with_bounding_boxes)

        detected_objects = detector.detectedObjects
        st.text('Recognized objects :')
        for i in detected_objects:
            st.text(i)

        st.text('The detected objects in ' + selected_language + ' is called :')
        for i in detected_objects:
            st.text(i + ' - ' + translate.translate(i))

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
