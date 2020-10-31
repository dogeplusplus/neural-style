start: style
	docker run -p 8500:8500 --mount type=bind,source=C:/Users/doge/github/neural-style/style,target=/models/style -e MODEL_NAME=style -t tensorflow/serving
