FROM tensorflow/tensorflow:2.10.1
COPY ./src/ .
COPY ./requirements.txt .
COPY docker/classifier-docker.yaml .
RUN apt-get update
RUN apt install  -y ffmpeg build-essential libdbus-glib-1-dev libgirepository1.0-dev tzdata libcairo2-dev libjpeg-dev python-cairo libhdf5-dev libopencv-dev cmake supervisor
RUN sed "s/tensorflow~=*/#tensorflow~=/" requirements.txt -i
RUN pip3 install --upgrade pip
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
