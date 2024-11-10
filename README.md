## Run Marathontime predict on a docker

1. Pull the docker image from docker hub
```docker pull guillermosb/marathontime-predict:latest```
2. Run the docker image
```docker run -p <PORT>:5000 guillermosb/marathontime-predict```
3. Predict the time
```curl -X POST http://localhost:<PORT>/predict -H "Content-Type: application/json" -d '{"wall_21": 2, "km_per_week": 21}'```
4. For effort: docker pull guillermosb/effort-predict:latest and run the docker image
```docker run -p <PORT>:8080 guillermosb/effort-predict```

