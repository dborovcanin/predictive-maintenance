FROM python:3.6.10-slim as builder
# FROM frolvlad/alpine-miniconda3:latest as builder
COPY ./main.py .
RUN pip3 install pandas scikit-learn
# RUN ./run.sh