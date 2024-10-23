FROM python:3.11

#set working directory
WORKDIR /app/backend

COPY . /app/backend

#install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#expose backend port

EXPOSE 9897

#run flask app
CMD ["gunicorn","--bind","0.0.0.0:9897","main:app"]