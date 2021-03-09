# importing libraries 
  
from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np 
import time

app = Flask(__name__)

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None

# Load names of classes and get random colors
classes = open('obj.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolo.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def load_image(frame):
    global img, img0, outputs, ln

    img0 = frame
    img = img0.copy()
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time() - t0

    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    outputs = np.vstack(outputs)

    post_process(img, outputs, 0.25)
    return img
    #cv.displayOverlay('window', f'forward propagation time={t:.3}')
    # cv.waitKey(0)

def post_process(img, outputs, conf):
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Create a VideoCapture object and read from input file 
cap = cv.VideoCapture('sample_input.mp4') 

# Check if camera opened successfully 
if (cap.isOpened()== False): 
	print("Error opening video file") 

def gen_frames():
	# Read until video is completed 
	while(cap.isOpened()): 
		
		# Capture frame-by-frame 
		ret, frame = cap.read() 
		if ret == True: 

			# Display the resulting frame 
			# cv.imshow('Frame', frame) 
			image = load_image(frame)
			s, buff = cv.imencode('.jpg', frame)
			frame = buff.tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

			# Press Q on keyboard to exit 
			if cv.waitKey(25) & 0xFF == ord('q'): 
				break

		# Break the loop 
		else: 
			break

	# When everything done, release 
	# the video capture object 
	cap.release() 

	# Closes all the frames 
	cv.destroyAllWindows() 

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
