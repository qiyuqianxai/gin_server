import docker 
import requests
import webbrowser
import os, sys
import random 
from time import sleep

class FaceBoard:
    def __init__(self, ip, port, mount, 
            open_browser=False,
            image="artifact.cloudwalk.work/rd_docker_dev/label-app/go", 
            conf=None):
        print("*** make sure mounted images dir and file dir ***")
        self.python_version = sys.version_info[0]
        self.browser = open_browser
        self.address = 'http://{}:{}'.format(ip, port)
        client = docker.from_env()
        try:
            _ = client.images.get(image)
        except docker.errors.ImageNotFound:
            client.images.pull(image)
        if not type(mount) in [list, tuple]:
            mount = [mount]
        volumes = {}
        for m in mount:
            volumes[m] = {'bind': m, 'mode': 'rw'}
        try:
            self.container = client.containers.run(image, 
                command="{}:{} {} {}".format(ip, port, mount, 'dummy_file'),
                volumes=volumes, ports={80: port}, auto_remove=True,
                detach=True)
        except Exception as e:
            print(e)
        else:
            sleep(3)
    
    def open_file(self, file):
        try:
            r = requests.post('{}/login'.format(self.address), data={'file':file, 'body':'', 'face':''})
            print("post return: ", r.status_code)
        except Exception as e:
            print(e)
            self.stop()
        else:
            base = os.path.basename(file)
            url = '{}/details/{}/off/1'.format(self.address, base)
            if not self.browser:
                print("please open the url manually: {}".format(url))
            else:
                webbrowser.get('chrome').open(url)
            if self.python_version == 2:
                input = raw_input
            while True:
                i = input("press q to quit")
                if i == "q":
                    self.stop()
                    break
    
    def stop(self):
        self.container.stop()
