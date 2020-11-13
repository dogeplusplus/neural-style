## Setup for both
- Run `firstTimeSetup.sh`
- Activate virtualenv `source venv/bin/activate`
- Download style transfer model `python download_model.py`

## Webcam Demo
- Run streamlit webcam application inside virtualenv `make webcam`


## Image Demo
- Pull tensorflow serving docker image and start tensorflow model server `make model-server-cpu`
- If you want to use GPU instead and have `nvidia-docker` then call `make model-server`
- In separate terminal call in virtualenv `make app`


## Dependencies
- Tensorflow serving apis
- Python 3.6+
- Docker
- Nvidia-docker
- tensorflow/serving:latest-gpu image
- tensorflow and tensorflow/serving repositories (for golang protobufs)