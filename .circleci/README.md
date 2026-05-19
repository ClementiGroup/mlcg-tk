# Docker image and CI
---
Here, we include an empty docker image based on `conda/miniconda3`. 
If you update the docker file, be sure to change the tag so that an image version change can be tracked.

# update docker image
```
name : jacopoventurin/mlcg_prebuilt_cpu_python_312:v*
```
For `v0.1`, after linking docker to `jacopoventurin` account on `DockerHub`, here is an example:
```
docker build  -t jacopoventurin/mlcg_prebuilt_cpu_python_312:v0.1 .
docker push jacopoventurin/mlcg_prebuilt_cpu_python_312:v0.1
```

To setup multiple arch images have a look at `https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/`.

Then do
```
docker buildx build --platform linux/amd64 -t jacopoventurin/mlcg_prebuilt_cpu_python_312:v0.1.1 --push .
```