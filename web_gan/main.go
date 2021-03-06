
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
	Fake_img_dir      string `json:"fake_img_dir"`
	User_img_dir      string `json:"user_img_dir"`
	Style_gan_message string `json:"style_gan_message"`
	Star_gan_message  string `json:"star_gan_message"`
	Static_file       string `json:"static_file"`
}

var config_file config

// send gan-nn param
type style_gan_message struct {
	Seed             string `json:"seed"`
	Age              string `json:"age"`
	Angle_horizontal string `json:"angle_horizontal"`
	Angle_pitch      string `json:"angle_pitch"`
	Beauty           string `json:"beauty"`
	Emotion_angry    string `json:"emotion_angry"`
	Emotion_disgust  string `json:"emotion_disgust"`
	Emotion_easy     string `json:"emotion_easy"`
	Emotion_fear     string `json:"emotion_fear"`
	Emotion_happy    string `json:"emotion_happy"`
	Emotion_sad      string `json:"emotion_sad"`
	Emotion_surprise string `json:"emotion_surprise"`
	Eyes_open        string `json:"eyes_open"`
	Face_shape       string `json:"face_shape"`
	Gender           string `json:"gender"`
	Glasses          string `json:"glasses"`
	Height           string `json:"height"`
	Race_black       string `json:"race_black"`
	Race_white       string `json:"race_white"`
	Race_yellow      string `json:"race_yellow"`
	Smile            string `json:"smile"`
	Width            string `json:"width"`
}
var style_gan_msg style_gan_message

type star_gan_message struct {
	User_img   string `json:"user_img"`
	Fake_img   string `json:"fake_img"`
	Blend_obj  string `json:"blend_obj"`
	Blend_type string `json:"blend_type"`
}
var star_gan_msg star_gan_message

func laod_config() (string,string,string,string,string) {
	_, err := os.Stat("config.json")
	if err != nil {
		return "", "", "","",""
	}
	file, err := os.Open("config.json")
	if err != nil {
		return "", "", "","",""
	}
	dco := json.NewDecoder(file)
	err = dco.Decode(&config_file)
	if err != nil {
		log.Println("????????????",err)
		return "", "", "","",""
	}
	return config_file.Fake_img_dir,config_file.User_img_dir,config_file.Style_gan_message,config_file.Star_gan_message,config_file.Static_file
}

func main() {
	// ?????????????????????
	_, user_img_pth, style_gan_message, star_gan_message,static_file := laod_config()

	//gin.SetMode(gin.DebugMode)
	r := gin.Default()

	// ????????????????????????
	r.Static("/static",static_file)

	//r.LoadHTMLGlob("static/*.html")
	r.LoadHTMLFiles(static_file+"/*.html")


	r.GET("/index", func(c *gin.Context) {
		// c.HTML(http.StatusOK, "index.html",nil)
		c.Redirect(http.StatusFound, "/static/index.html")
	})

	r.GET("/animal_gan", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/static/animal_gan.html")
	})

	r.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusFound,"/index")
	})

	// ??????????????????
	r.GET("/get_base_info", func(c *gin.Context) {
		_, err := os.Stat(user_img_pth)
		var user_imgs []string
		if err == nil {
			f_list,err := ioutil.ReadDir(user_img_pth)
			if err!=nil{
				log.Fatal(err)
			} else {
				for _,f := range f_list{
					user_imgs = append(user_imgs, f.Name())
				}
			}
		}
		if os.IsNotExist(err) {
			err := os.Mkdir(user_img_pth, os.ModePerm)
			if err != nil {
				return 
			}
		}
		c.JSON(200,gin.H{
			"img_list":user_imgs,
		})
	})

	// blend images
	r.POST("/convert_img/", func(c *gin.Context) {
		err := c.BindJSON(&star_gan_msg)
		if err != nil {
			return
		}
		//log.Printf("%v",&msg)

		// ???json??????
		_, err = os.Stat(star_gan_message)
		var file *os.File
		if err == nil {
			file, err = os.OpenFile(star_gan_message,os.O_WRONLY|os.O_TRUNC,0666)
			if err != nil {
				log.Println(err)
			}
		}else {
			file, err = os.Create(star_gan_message)
			if err != nil {
				log.Println(err)
			}
		}
		enc := json.NewEncoder(file)
		err = enc.Encode(star_gan_msg)
		if err != nil {
			log.Println(err)
		}
		err = file.Close()
		if err != nil {
			log.Println(err)
		}
		time.Sleep(time.Duration(3)*time.Second);
		//for i:=0;i<100;i++{
		//	blend_img := "/static/blend.jpg"
		//	_, err :=os.Stat(blend_img)
		//	if err == nil {
		//		log.Println("generate dest image success!")
		//		break
		//	}
		//	time.Sleep(time.Duration(1)*time.Second);
		//}
		c.JSON(200,gin.H{})
	})

	// upload image
	r.POST("/upload_file/", func(c *gin.Context) {
		upfile, err := c.FormFile("file")
		if err != nil {
			return
		}
		img_name := upfile.Filename
		log.Println(img_name)
		if strings.HasSuffix(img_name,".jpeg") || strings.HasSuffix(img_name,".jpg") || strings.HasSuffix(img_name,".png"){
			save_pth := path.Join(user_img_pth,img_name)
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

	// generate image
	r.POST("/generate_img/", func(c *gin.Context) {
		err := c.BindJSON(&style_gan_msg)
		if err != nil {
			return
		}
		//log.Printf("%v",&msg)

		// ???json??????
		_, err = os.Stat(style_gan_message)
		var file *os.File
		if err == nil {
			file, err = os.OpenFile(style_gan_message,os.O_WRONLY|os.O_TRUNC,0666)
			if err != nil {
				log.Println(err)
			}
		}else {
			file, err = os.Create(style_gan_message)
			if err != nil {
				log.Println(err)
			}
		}
		enc := json.NewEncoder(file)
		err = enc.Encode(style_gan_msg)
		if err != nil {
			log.Println(err)
		}
		err = file.Close()
		if err != nil {
			log.Println(err)
		}
		time.Sleep(time.Duration(3)*time.Second);
		c.JSON(200,gin.H{})
	})

	err := r.Run(":43476")
	if err != nil {
		return 
	}
}




