FROM python:3

RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN python -m pip install --upgrade pip
RUN pip install pyaudio
