model-server: style
	docker run --gpus all -p 8500:8500 --mount type=bind,source=/home/albert/github/neural-style/style,target=/models/style -e MODEL_NAME=style -t tensorflow/serving:latest-gpu

app:
	exec streamlit run streamlit_app.py
