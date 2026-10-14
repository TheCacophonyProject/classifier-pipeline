FROM tensorflow/tensorflow:2.21.0
COPY ./src/ .
COPY ./requirements.txt .
COPY docker/classifier-docker.yaml .
COPY docker/latestmodel.py .

# FIX: Switch to a fast cloud mirror & combine update/install to prevent stalls
RUN sed -i 's|http://ubuntu.com|http://ubuntu.com|g' /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libdbus-glib-1-dev \
    libgirepository1.0-dev \
    tzdata \
    libcairo2-dev \
    libjpeg-dev \
    python3-gi-cairo \
    libhdf5-dev \
    libopencv-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN sed "s/tensorflow~=*/#tensorflow~=/" requirements.txt -i
#pybgs library doesn't work with python3.11
# RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
# RUN apt install -y python3.10-dev
RUN python3 get-pip.py
RUN pip3 install --upgrade pip
RUN pip3 install cmake
RUN pip3 install -r requirements.txt
RUN pip3 install ultralytics==8.4.166
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

RUN mkdir -p /var/log/supervisor
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p /etc/cacophony
RUN cp classifier-docker.yaml /etc/cacophony/classifier.yaml
RUN touch /etc/cacophony/classifier
RUN python3 latestmodel.py


RUN wget https://github.com/TheCacophonyProject/AI-Model/releases/download/yolo-v1.0/model.tar
RUN mkdir /etc/cacophony/models/yolo
RUN tar xzvf model.tar -C /etc/cacophony/models/yolo --strip-components=1
RUN rm model.tar

RUN pip install --no-cache-dir --upgrade keras==3.14.1

COPY docker/ready.sh .
WORKDIR /
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf", "-n"]
