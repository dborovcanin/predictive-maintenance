FROM python:3.6.10-slim as builder
# FROM frolvlad/alpine-miniconda3:latest as builder
COPY ./train.py .
COPY ./transform.py .
COPY ./train.sh .
COPY ./transform.sh .
RUN pip3 install pandas scikit-learn