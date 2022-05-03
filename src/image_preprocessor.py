import cv2
import imutils
import mediapipe as mp
import numpy as np
import streamlit as st

# image preprocessing setup

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                            min_detection_confidence=0.5)


@st.cache()
def image_preprocessor(uploaded_image):
    processed_image = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = processed_image.shape

    results = face_mesh.process(processed_image)
    if not results.multi_face_landmarks:
        st.error("No face detected in the picture")
        return None
    else:
        face_landmarks = results.multi_face_landmarks[0]

        LEFT_EYE_CORNER_INDEX = 33
        LEFT_EYE_POSITION = (
            int(face_landmarks.landmark[LEFT_EYE_CORNER_INDEX].x * image_width),
            int(face_landmarks.landmark[LEFT_EYE_CORNER_INDEX].y * image_height))

        LEFT_EYE_CORNER_INDEX = 263
        RIGHT_EYE_POSITION = (
            int(face_landmarks.landmark[LEFT_EYE_CORNER_INDEX].x * image_width),
            int(face_landmarks.landmark[LEFT_EYE_CORNER_INDEX].y * image_height))

        EYES_CENTER = (
            (LEFT_EYE_POSITION[0] + RIGHT_EYE_POSITION[0]) // 2,
            (LEFT_EYE_POSITION[1] + RIGHT_EYE_POSITION[1]) // 2)

        dX = LEFT_EYE_POSITION[0] - RIGHT_EYE_POSITION[0]
        dY = LEFT_EYE_POSITION[1] - RIGHT_EYE_POSITION[1]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        center = (image_width // 2, image_height // 2)
        rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cosofRotationMatrix = np.abs(rotationMatrix[0][0])
        sinofRotationMatrix = np.abs(rotationMatrix[0][1])
        newImageWidth = int((image_height * sinofRotationMatrix) + (image_width * cosofRotationMatrix))
        newImageHeight = int((image_height * cosofRotationMatrix) + (image_width * sinofRotationMatrix))
        rotationMatrix[0][2] += (newImageWidth / 2) - center[0]
        rotationMatrix[1][2] += (newImageHeight / 2) - center[1]

        processed_image = cv2.warpAffine(processed_image, rotationMatrix, (newImageWidth, newImageHeight),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        image_height, image_width, _ = processed_image.shape

        results = face_detection.process(processed_image)
        if not results.detections:
            st.error("No face detected in the picture")
            return None
        else:
            detection = results.detections[0]  # we pick the most probable result

            bb_data = detection.location_data.relative_bounding_box
            bounding_box = (
                int(bb_data.xmin * image_width - 0.4 * bb_data.width * image_width),
                int(bb_data.ymin * image_height - 0.4 * bb_data.width * image_width * 1.8 * 218 / 178),
                int(bb_data.width * image_width * 1.8),
                int(bb_data.width * image_width * 1.8 * 218 / 178))

            # add padding for cropping out of bounds of the image
            padding = max(image_height, image_width)

            processed_image = cv2.copyMakeBorder(processed_image, padding, padding, padding, padding,
                                                 cv2.BORDER_REPLICATE)

            bounding_box = bounding_box[0] + padding, bounding_box[1] + padding, bounding_box[2], bounding_box[3]

            # cropps and resizes to final dimentions
            processed_image = processed_image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                              bounding_box[0]:bounding_box[0] + bounding_box[2]]
            processed_image = imutils.resize(processed_image, width=178,
                                             height=218)  # resizes to either 178 or 218 (with high quality)
            processed_image = cv2.resize(processed_image,
                                         (178, 218))  # forces resize to 178x218, but if used alone lowers quality
            return processed_image
