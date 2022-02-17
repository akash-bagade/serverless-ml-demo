FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Calcutta

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         sudo \
         git \
         python3-opencv \
         python3-dev \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip3 install boto3

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY container /opt/program
WORKDIR /opt/program