import os
import json

json_pth = 'D:/annotation_val.odgt'
with open(json_pth,"r",encoding="utf-8")as f:
    lines = f.readlines()
new_f = ""
for line in lines:
    line = line.replace("\n","")
    img_label_info = json.loads(line)
    print(img_label_info)
    new_line = ""
    img_pth = "dev/"+img_label_info["ID"]+".jpg"
    new_line += img_pth + " "
    for boxinfo in img_label_info["gtboxes"]:
        x1,y1,x2,y2 = boxinfo["hbox"]
        x2 += x1
        y2 += y1
        new_line += str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+",0 "
        x1, y1, x2, y2 = boxinfo["fbox"]
        x2 += x1
        y2 += y1
        new_line += str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ",1 "
    new_line = new_line.strip()+"\n"
    new_f += new_line

with open("dev.txt","w",encoding="utf-8")as f:
    f.write(new_f)
