## Webcam Demo
- Run `firstTimeSetup.sh`
- Activate virtualenv `source venv/bin/activate`
- Run streamlit webcam application `streamlit run webcam_stream.py`


## Image Demo
- Run `firstTimeSetup.sh`
- Activate virtualenv `source venv/bin/activate`
- Pull tensorflow serving docker image and start tensorflow model server `make model-server`
- In separate terminal call `make app`


## Dependencies
- Tensorflow serving apis
- Python 3.6+
- Docker
- Nvidia-docker
- tensorflow/serving:latest-gpu image
- Go
- gocv
- tensorflow and tensorflow/serving repositories (for golang protobufs)
- golang protobuf library
- protobuf compiler