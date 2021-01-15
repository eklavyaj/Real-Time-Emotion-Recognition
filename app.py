from model import *
import streamlit as st
from PIL import Image, ImageEnhance
from mtcnn import MTCNN

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
	
	st.title("Emotion Classifier")
	st.subheader("Feel the emotion, I'll tell you what it is.")

	activites = ["Detection", "About"]
	choice = st.sidebar.selectbox("Select Activity",activites)
	if choice == "Detection":
		# st.subheader("Emotion Classifier")
		upload_file = st.file_uploader("Upload an image", type=("png", "jpg", "jpeg"))
		our_image = 0
		if upload_file is not None:
			our_image = Image.open(upload_file)
			w , h = our_image.size
			nw = min(400,w)
			nh = int(nw*h/w)
			our_image = our_image.resize((nw,nh))
			st.text("Original Image")
			st.image(our_image)

		enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])

		out_image = our_image
		if enhance_type == "Gray-Scale":
			if upload_file is not None:
				new_img = np.array(our_image.convert('RGB'))
				img = cv2.cvtColor(new_img, 1)
				out_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				out_image = Image.fromarray(out_image)
				st.text("Gray-Scale Image")
				st.image(out_image)
			else:
				st.write("Please Upload an Image")

		elif enhance_type == 'Contrast':

			if upload_file is not None:
				c_rate = st.sidebar.slider("Contrast",0.5,3.5)
				enhancer = ImageEnhance.Contrast(our_image)
				out_image = enhancer.enhance(c_rate)
				st.text("Image with varying contrast")
				st.image(out_image)

			else :
				st.write("Please Upload an Image")				

		elif enhance_type == 'Brightness':

			if upload_file is not None:
				c_rate = st.sidebar.slider("Brightness",0.5,3.5)
				enhancer = ImageEnhance.Brightness(our_image)
				out_image = enhancer.enhance(c_rate)
				st.text("Image with varying brightness")
				st.image(out_image)

			else :
				st.write("Please Upload an Image")

		elif enhance_type == 'Blurring':

			if upload_file is not None:
				new_img = np.array(our_image.convert('RGB'))
				blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
				img = cv2.cvtColor(new_img,1)
				out_image = cv2.GaussianBlur(img,(11,11),blur_rate)
				out_image = Image.fromarray(out_image)
				st.text("Image with varying blur effect")
				st.image(out_image)

			else :
				st.write("Please Upload an Image")



		button = st.sidebar.button('Process Image')
		if upload_file is not None and button:
			
			st.write("Detection")

			face_detector_mtcnn = MTCNN()
			# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			model = FacialExpressionModel('model/model.json', 'model/model_weights.h5')
			frame = np.array(out_image)

			# frame = cv2.resize(frame, (400,400))
			if enhance_type == 'Gray-Scale':
				gray = frame
			else:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# faces = face_cascade.detectMultiScale(gray, 1.3,4)
			if enhance_type != 'Gray-Scale':
				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			else:
				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

			dict_faces = face_detector_mtcnn.detect_faces(frame_rgb) 
			faces =[]
			for dict_face in dict_faces:
				x,y,w,h = dict_face['box'][:]
				faces.append([x,y,w,h])
			# print(faces)
			st.write("Number of faces detected - {}.".format(len(faces)))

			i = 1
			for (x,y,w,h) in faces:
				face = gray[y:y+h , x:x+h]
				roi = cv2.resize(face, (64,64))

				pred, confidence = model.predict_emotion(roi[np.newaxis, :, : , np.newaxis])

				cv2.putText(frame, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (69,143,46), 2)
				cv2.rectangle(frame, (x,y), (x+w, y+h) , (255,255,0),2)
				st.write(str(i) + ". "+pred.capitalize())
				i+=1

			st.image(frame)
			# cv2.imshow("detection", frame)
			# cv2.waitKey(0)

	else:
		st.write("This is an attempt to perform emotion recognition from facial expressions. There is added functionality to format the image as required. The model will run on the formatted image.")

		st.write("~ A project by Eklavya Jain")

if __name__ == "__main__":
	main()

