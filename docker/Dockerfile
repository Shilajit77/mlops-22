#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirement.txt /exp/requirement.txt
COPY ./docker_qstn.py /docker_qstn.py
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
#EXPOSE 5002
#CMD ["python3", "./app.py"]