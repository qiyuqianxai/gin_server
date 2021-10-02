
package main

import (
	//"awesomeProject/utils"
	"encoding/json"
	"github.com/gin-gonic/gin"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"strings"
	"time"
)

type config struct {
	Dest_img_dir  string `json:"dest_img_dir"`
	Src_img_dir   string `json:"src_img_dir"`
	Message_json string `json:"message_json"`
	Static_file  string `json:"static_file"`
}

var config_file config

// send gan-nn param
type message struct {
	Src_img      string `json:"src_img"`
	Gan_gender string `json:"gan_gender"`
	Gan_eyes_color string `json:"gan_eyes_color"`
	Gan_hair_color string `json:"gan_hair_color"`
}

var msg message


func laod_config() (string,string,string,string) {
	_, err := os.Stat("config.json")
	if err != nil {
		return "", "", "",""
	}
	file, err := os.Open("config.json")
	if err != nil {
		return "", "", "",""
	}
	dco := json.NewDecoder(file)
	err = dco.Decode(&config_file)
	if err != nil {
		log.Println("解析错误",err)
		return "", "", "",""
	}
	return config_file.Dest_img_dir,config_file.Src_img_dir,config_file.Message_json,config_file.Static_file
}


func main() {
	// 读取配置的路径
	dest_img_dir, src_img_pth, message_json, static_file := laod_config()

	//gin.SetMode(gin.DebugMode)
	r := gin.Default()

	// 定义静态文件映射
	r.Static("/static",static_file)

	//r.LoadHTMLGlob("static/*.html")
	r.LoadHTMLFiles(static_file+"/*.html")


	r.GET("/index", func(c *gin.Context) {
		// c.HTML(http.StatusOK, "index.html",nil)
		c.Redirect(http.StatusFound, "/static/index.html")
	})

	r.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusFound,"/index")
	})

	// 获取基本信息
	r.GET("/get_base_info", func(c *gin.Context) {
		_, err := os.Stat(src_img_pth)
		var src_imgs []string
		if err == nil {
			f_list,err := ioutil.ReadDir(src_img_pth)
			if err!=nil{
				log.Fatal(err)
			} else {
				for _,f := range f_list{
					src_imgs = append(src_imgs, f.Name())
				}
			}
		}
		if os.IsNotExist(err) {
			err := os.Mkdir(src_img_pth, os.ModePerm)
			if err != nil {
				return 
			}
		}
		c.JSON(200,gin.H{
			"img_list":src_imgs,
		})
	})

	// 更新本地文件
	r.POST("/convert_img/", func(c *gin.Context) {
		err := c.BindJSON(&msg)
		if err != nil {
			return
		}
		//log.Printf("%v",&msg)

		// 写json文件
		_, err = os.Stat(message_json)
		var file *os.File
		if err == nil {
			file, err = os.OpenFile(message_json,os.O_WRONLY|os.O_TRUNC,0666)
			if err != nil {
				log.Println(err)
			}
		}else {
			file, err = os.Create(message_json)
			if err != nil {
				log.Println(err)
			}
		}
		enc := json.NewEncoder(file)
		err = enc.Encode(msg)
		if err != nil {
			log.Println(err)
		}
		err = file.Close()
		if err != nil {
			log.Println(err)
		}

		for i:=0;i<100;i++{
			dest_img := strings.Replace(msg.Src_img,".jpg","",-1)
			dest_img = strings.Replace(dest_img,".jpeg","",-1)
			dest_img = strings.Replace(dest_img,".png","",-1)
			gender := msg.Gan_gender
			eyes_color := msg.Gan_eyes_color
			hair_color := msg.Gan_hair_color
			dest_img = dest_img + "_" + gender + "_" + eyes_color + "_" + hair_color +".jpg"
			_, err :=os.Stat(path.Join(dest_img_dir,dest_img))
			log.Println(path.Join(dest_img_dir,dest_img))
			if err == nil {
				log.Println("generate dest image success!")
				break
			}
			time.Sleep(time.Duration(1)*time.Second);
		}
		c.JSON(200,gin.H{})
	})

	// 上传文件
	r.POST("/upload_file/", func(c *gin.Context) {
		upfile, err := c.FormFile("file")
		if err != nil {
			return
		}
		img_name := upfile.Filename
		log.Println(img_name)
		if strings.HasSuffix(img_name,".jpeg") || strings.HasSuffix(img_name,".jpg") || strings.HasSuffix(img_name,".png"){
			save_pth := path.Join(src_img_pth,img_name)
			_, err = os.Stat(save_pth)
			if !os.IsNotExist(err) {
				err := os.Remove(save_pth)
				if err != nil {
					return
				}
			}

			err = c.SaveUploadedFile(upfile, save_pth)
			if err != nil {
				return
			}
			c.JSON(200,gin.H{})
		} else{
			c.JSON(304,gin.H{})
		}
	})

	err := r.Run(":43476")
	if err != nil {
		return 
	}
}




