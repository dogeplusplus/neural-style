CURRENT_DIR = $(shell pwd)

model-server: style
	docker run --gpus all -p 8500:8500 --mount type=bind,source=${CURRENT_DIR}/style,target=/models/style -e MODEL_NAME=style -t tensorflow/serving:latest-gpu

model-server-cpu: style
	docker run -p 8500:8500 --mount type=bind,source=${CURRENT_DIR}/style,target=/models/style -e MODEL_NAME=style -t tensorflow/serving

app:
	exec streamlit run streamlit_app.py

webcam:
	exec streamlit run webcam_stream.py
