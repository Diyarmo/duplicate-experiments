FROM docker.repos.balad.ir/nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04


RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y openssh-client && \
    apt-get install -y python3.8 && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.8-venv


COPY requirements.txt /home/

WORKDIR /home

RUN python3.8 -m venv /venv

ENV PATH=/venv/bin:$PATH

RUN python -m pip install --upgrade pip

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /home/

ARG PROJECT_GIT_PATH="divar/review/bots/bots-dev/duplicate-bot"
ARG GIT_EMAIL="diyar.mohammadi@divar.ir"
ARG GIT_USERNAME="airflow-training-pipeline-user"

RUN git config user.email ${GIT_EMAIL}
RUN git config user.name ${GIT_USERNAME}
RUN git remote set-url origin https://git.cafebazaar.ir/${PROJECT_GIT_PATH}

ENTRYPOINT ["bash"]
# CMD bash run_pipeline.sh
