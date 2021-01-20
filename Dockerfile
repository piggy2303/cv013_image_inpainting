FROM nvidia/cuda:11.1-base-ubuntu20.04
FROM pytorch/pytorch

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install python3-pip python3-dev libgl1-mesa-glx -y

RUN pip3 install --upgrade pip

RUN pip install Flask
RUN pip install Flask-Cors==1.10.3
RUN pip install gevent
# RUN pip install matplotlib
RUN pip install opencv-python
# RUN pip install scikit-image==0.14.2
# RUN pip install scipy
RUN pip install googledrivedownloader

ADD . /app
WORKDIR /app

RUN python3 download_models.py

CMD python3 wsgi.py
