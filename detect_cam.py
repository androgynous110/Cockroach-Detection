import cv2
from inference_sdk import InferenceHTTPClient

# Initialize the InferenceHTTPClient with your API URL and API key
CLIENT = InferenceHTTPClient(
    api_url="GET YOUR URL API",
    api_key="GET YOUR API KEY"

    # to get your api key and url go to this link: https://universe.roboflow.com/school-project-cwbwv/esp3902
)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Read frames from the webcam capture
    ret, frame = cap.read()

    # Rotate the frame by 180 degrees horizontally to correct the upside-down orientation
    frame = cv2.flip(frame, 0)

    result = CLIENT.infer(frame, model_id="esp3902/1")  # Make inference request using the initialized client

    labels = []
    rectangles = []
    length = 50

    for prediction in result['predictions']:
        label = prediction['class']
        bbox = {
            'xmin': int(prediction['x']),
            'ymin': int(prediction['y']),
        }

        labels.append(label)
        rectangles.append(bbox)
        
    for rectangle in rectangles:
        xmin = rectangle['xmin']
        ymin = rectangle['ymin']
        
        # Calculate the center of the line
        line_center_x = xmin + (length // 2)
        line_center_y = ymin
        
        # Draw the radar lines around the center
        cv2.line(frame, (line_center_x - length, line_center_y), (line_center_x + length, line_center_y), (0, 255, 0), 2)
        cv2.line(frame, (line_center_x, line_center_y - length), (line_center_x, line_center_y + length), (0, 255, 0), 2)

        # Display the label at the center
        cv2.putText(frame, 'Cockroach', (line_center_x + 10, line_center_y + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()