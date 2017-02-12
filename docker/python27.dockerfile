FROM tensorflow/tensorflow:0.12.1

RUN apt-get update
RUN apt-get -y install git

RUN apt-get update
RUN apt-get -y install git


RUN pip install pdoc
RUN pip install mako
RUN pip install markdown
RUN pip install pytest
RUN pip install pytest-sugar