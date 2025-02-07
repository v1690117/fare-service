## Face Recognition Service

Service that does only one thing. It can take photo and say if there is person on the photo who is known to the
server.  
To identify whether the person is known it uses prepared model.  
Tool to prepare model based on your data is present too.

The service is not about ML. It just uses it.

### Development

Install dependencies (`This could require cmake, install it if needed`).

```cmd
pip install requirements.txt
```

Run development server.

```cmd
python server.py
```

Make a http request (Intellij IDEA Http Client sample).

```http request
POST http://localhost:8080/recognize
Content-Type: multipart/form-data; boundary=WebAppBoundary

--WebAppBoundary
Content-Disposition: form-data; name="file"; filename="picture.jpg"
Content-Type: application/json

< ./data/tests/many.jpg
--WebAppBoundary--
```

### Train

You need photos of the person being placed in `./data/samples`.  
There must be only one person on each photo.  
Call `train.py` script to prepare model.

```cmd
python train.py
```

New model will be saved in file `model.pkl`.

### Testing

Put your test photos into `./data/tests` and call the `test.py` script.

```cmd
python test.py
```

It will show which photos contain the person.

### Deployment

Just build the dockerfile.

```cmd
docker build -it fare-service .
```

Run the container.

```cmd
docker run -it -d -p 8080:8080 fare-service
```
