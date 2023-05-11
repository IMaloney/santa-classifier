FROM tensorflow/tensorflow:latest

WORKDIR /project

COPY . /project

RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN chmod +x /project/code/train_main.py

ENTRYPOINT ["python", "code/train_main.py", "-g", "-r", "<SET ME>"]