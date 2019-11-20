from model.inception_resnet_v1 import *
from utility import *

def real_time_recog():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    metric = "euclidean"
    if metric == "cosine":
	    threshold = 0.10
    else:
	    threshold = 0.50
    known_faces = "known_faces/"
    people = dict()

    # model = model_from_json(open("./model/facenet_model.json", "r").read())
    # print("model built")
    # model = InceptionResNetV1()
    model = load_model('./model_keras/facenet_keras.h5')
    model.load_weights('./model_keras/facenet_keras.h5')
    model._make_predict_function()
    print("weights loaded")

    for file in listdir(known_faces):
	    print(file)
	    known_face, extension = file.split(".")
	    img = preprocess_image('known_faces/%s.jpg' % (known_face))
	    representation = model.predict(img)[0,:]
	    people[known_face] = l2_normalize(representation)
	
    print("known_face representations retrieved successfully")
    cap = cv2.VideoCapture(0) #webcam

    while(True):
	    ret, img = cap.read()
	    faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	    for (x,y,w,h) in faces:
		    if w > 130: #discard small detected faces
			    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 255, 0), 2) #draw rectangle to main image
			
			    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			    detected_face = cv2.resize(detected_face, (160, 160)) #resize to 224x224
			
			    img_pixels = image.img_to_array(detected_face)
			    img_pixels = np.expand_dims(img_pixels, axis = 0)
			    #known_face dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
			    img_pixels /= 127.5
			    img_pixels -= 1
			
			    captured_representation = model.predict(img_pixels)[0,:]
			    captured_representation = l2_normalize(captured_representation)
			    distances = []
			
			    for i in people:
				    known_face_name = i
				    source_representation = people[i]
				    if metric == "cosine":
					    distance = findCosineDistance(captured_representation, source_representation)
				    elif metric == "euclidean":
					    distance = findEuclideanDistance(captured_representation, source_representation)
				    print(known_face_name,": ",distance)
				    distances.append(distance)
			
			    label_name = 'Unknown'
			    index = 0
			    for i in people:
				    known_face_name = i
				    if index == np.argmin(distances):
					    if distances[index] <= threshold:
						    print("detected: ",known_face_name)
						
						    #label_name = "%s (distance: %s)" % (known_face_name, str(round(distance,2)))
						    if metric == "euclidean":
							    similarity = 100 + (20 - distance)
						    elif metric == "cosine":
							    similarity = 100 + (40 - 100*distance)
						
						    if similarity > 99.99: similarity = 99.99
						
						    label_name = "%s (%s%s)" % (known_face_name, str(round(similarity,2)), '%')
						
						    break
					
				    index = index + 1
			
			    # cv2.putText(img, label_name, (int(x+w+15), int(y-64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (67,67,67), 2)
			    cv2.putText(img, label_name, (int(x+w+15), int(y-64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
		
			    #connect face and text
			    cv2.line(img,(x+w, y-64),(x+w-25, y-64),(255, 255, 0),1)
			    cv2.line(img,(int(x+w/2),y),(x+w-25,y-64),(255, 255, 0),1)
			
	    cv2.imshow('img',img)
	
	    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		    break
	
    #kill open cv things		
    cap.release()
    cv2.destroyAllWindows()


real_time_recog()    