FROM tensorflow/tensorflow:2.14.0
COPY ./src/ .
COPY ./requirements.txt .
COPY docker/classifier-docker.yaml .
RUN apt-get update
RUN apt install  -y python3.10-dev ffmpeg build-essential libdbus-glib-1-dev libgirepository1.0-dev tzdata libcairo2-dev libjpeg-dev python3-gi-cairo libhdf5-dev libopencv-dev supervisor
RUN sed "s/tensorflow~=*/#tensorflow~=/" requirements.txt -i
#pybgs library doesn't work with python3.11
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN apt install -y python3.10-dev
RUN python3 get-pip.py
RUN pip3 install --upgrade pip
RUN pip3 install cmake
RUN pip3 install numpy
RUN pip3 install -r requirements.txt


RUN mkdir -p /var/log/supervisor
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir /etc/cacophony/models/inc3 -p

RUN cp classifier-docker.yaml /etc/cacophony/classifier.yaml
RUN touch /etc/cacophony/classifier
RUN pip install gdown
RUN gdown --fuzzy "https://drive.google.com/file/d/1DSltJvJc_qj7Eyh6CDWsshOf8gqAD0aM/view?usp=sharing" -O thermal-model.tar
RUN tar xzvf thermal-model.tar -C /etc/cacophony/models/inc3 --strip-components=1
RUN rm thermal-model.tar

WORKDIR /
