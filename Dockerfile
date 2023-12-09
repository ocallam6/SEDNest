FROM continuumio/miniconda3

RUN mkdir -p SEDNest

COPY . /SEDNest
WORKDIR /SEDNest

RUN conda env update --file environment.yml

RUN echo "conda activate SEDNest" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
