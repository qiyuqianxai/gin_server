# How to run 

you can either use faceboard python library or deploy a docker container as service 


## 1. Use faceboard python library
```bash
pip install dist/faceboard-0.1-py2.py3-none-any.whl
```

```python
from faceboard import FaceBoard
"""
Parameters
----------
ip : the ip address of the host
file : path of the <image, label, meta> file
mount : the host path which has images
open_browser : if open new tab to show faceboard or not
"""
fb = FaceBoard('10.128.128.88', '6789', "/data")
fb.open_file("/data/cluster_rst/ijbc_align")

```
If use faceboard inside docker container please mount ```-v /var/run/docker.sock:/var/run/docker.sock``` 

## 2. Docker service
### Build docker image

```bash
docker build -t go-label:latest .
```

### Pull image from dockerhub

```bash
docker pull artifact.cloudwalk.work/rd_docker_dev/label-app/go:latest
```

## Start service

```bash
port=...       # the host port
address=...    # the host address 
docker run --rm --name go-label -v $(pwd):/ssd -p $port:80 artifact.cloudwalk.work/rd_docker_dev/label-app/go:latest $address:$port /ssd
```

## 3. About data

After started the container, 5 folders will be made in `$(pwd)` if not exists:

**folder** | **purpose**
--- | ---
log | saving nginx logs
cluster_rst | go-label loads face clustering result files from here
body_rst | go-label loads body clustering result files from here
db | databases for body details data 
imgs | go-label loads local images from here

### Clustering result files
#### Using ***starbox id***, they look like:
```bash
1517,36a7ee05d49c90ad 0
1522,3716b21dc0159b23 0
1521,36f76cb53804bc7f 1
1514,368f8e44fb47a834 1
```
#### Using ***local files***:
```bash
$dir_name/1.jpg 0
$dir_name/2.jpg 0
$dir_name/3.jpg 1
$dir_name/4.jpg 1
```
If loading images from local, please mount `$dir_name`.

#### Show meta infomation:
```bash
path            label  color  meta ...
$dir_name/1.jpg   0      0     ...
$dir_name/2.jpg   0      1     ...
$dir_name/3.jpg   0      2     ...
$dir_name/4.jpg   0      3     ...
```
for the color column:
0 is transparent, (1, 2, 3) is rgb respectively, bigger number is random generated color