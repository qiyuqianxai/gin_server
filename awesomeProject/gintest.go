
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
	"strconv"
	"strings"
	"time"
)

type config struct {
	Predict_dir  string `json:"predict_dir"`
	Videos_pth   string `json:"videos_pth"`
	Message_json string `json:"message_json"`
	Static_file  string `json:"static_file"`
}

var config_file config

type message struct {
	Current_id int `json:"id"`
	Play_video 		bool
	Video_name      string `json:"video_name"`
	// 需要的ai功能
	Req_face_detect    bool `json:"req_face_detect"`
	Req_body_detect    bool `json:"req_body_detect"`
	Req_body_keypoints bool `json:"req_body_keypoints"`
	Req_face_keypoints bool `json:"req_face_keypoints"`
	Req_expression_reg bool `json:"req_expression_reg"`
	Req_action_reg     bool `json:"req_action_reg"`
	Person_track       bool `json:"person_track"`
	Req_attention_reg  bool `json:"req_attention_reg"`
}

var msg message

type boxinfo struct {
	Box                []float32     `json:"box"`
	Keypoints          [][]float32   `json:"keypoints"`
	Adjacent_keypoints [][][]float32 `json:"adjacent_keypoints"`
}
type label_result struct {
	Image_name string    `json:"image_name"`
	Body_info  []boxinfo `json:"body_info"`
}

type data_analysis_res struct {
	Action_analysis     [][]string `json:"action_analysis"`
	Expression_analysis [][]string `json:"expression_analysis"`
	Attention_analysis  [][]string `json:"attention_analysis"`
	Person_num          int        `json:"person_num"`
}


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
		println("解析错误",err)
		return "", "", "",""
	}
	return config_file.Predict_dir,config_file.Videos_pth,config_file.Message_json,config_file.Static_file
}


func main() {
	// 读取配置的路径
	predict_dir, videos_pth, message_json, static_file := laod_config()

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
		_, err := os.Stat(videos_pth)
		var video_names []string
		video_names = append(video_names,"capture")
		if err == nil {
			f_list,err := ioutil.ReadDir(videos_pth)
			if err!=nil{
				log.Fatal(err)
			} else {
				for _,f := range f_list{
					video_names = append(video_names, f.Name())
				}
			}
		}
		if os.IsNotExist(err) {
			err := os.Mkdir(videos_pth, os.ModePerm)
			if err != nil {
				return 
			}
		}
		c.JSON(200,gin.H{
			"video_list":video_names,
		})
	})

	// 更新本地文件
	r.POST("/update_play_info", func(c *gin.Context) {
		msg.Play_video = true
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
				println(err)
			}
		}else {
			file, err = os.Create(message_json)
			if err != nil {
				println(err)
			}
		}
		enc := json.NewEncoder(file)
		err = enc.Encode(msg)
		if err != nil {
			println(err)
		}
		err = file.Close()
		if err != nil {
			println(err)
		}

		// 获取max_id
		current_video := msg.Video_name
		max_id := -1
		_, err = os.Stat(path.Join(predict_dir, current_video))
		if err != nil {
			println(err)
		}
		dir, err := ioutil.ReadDir(path.Join(predict_dir,current_video))
		if err != nil {
			println(err)
		}

		for _,f := range dir{
			if strings.HasSuffix(f.Name(),"_predict.json"){
				max_id++
			}
		}
		if max_id - msg.Current_id < 10{
			time.Sleep(5);
		}
		c.JSON(200,gin.H{"max_id":max_id})
	})

	// 播放结束
	r.GET("/play_end/", func(c *gin.Context) {
		msg.Play_video = false
		file, err := os.OpenFile(message_json,os.O_WRONLY|os.O_TRUNC,0666)
		if err != nil {
			println(err)
		}

		enc := json.NewEncoder(file)
		err = enc.Encode(msg)
		if err != nil {
			println(err)
		}
		err = file.Close()
		if err != nil {
			println(err)
		}
		c.JSON(200,gin.H{})
	})

	// 上传文件
	r.POST("/upload_file/", func(c *gin.Context) {
		upfile, err := c.FormFile("file")
		if err != nil {
			return
		}
		video_name := upfile.Filename
		log.Println(video_name)
		if strings.HasSuffix(video_name,".mp4") || strings.HasSuffix(video_name,".avi") || strings.HasSuffix(video_name,".flv"){
			save_pth := path.Join(videos_pth,video_name)
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

	// 获取数据分析的数据
	r.POST("/get_analysis_data/", func(c *gin.Context) {
		data_json := make(map[string]interface{}) //注意该结构接受的内容
		err := c.BindJSON(&data_json)
		if err != nil {
			return 
		}
		//log.Printf("%v",&data_json)
		img_id := int(data_json["id"].(float64))
		current_video := data_json["video"].(string)
		json_pth := path.Join(predict_dir,current_video,strconv.Itoa(img_id)+".json")
		_, err = os.Stat(json_pth)
		if os.IsNotExist(err){
			// c.JSON(200,gin.H{})
			log.Println(err)
			return
		}

		var json_file *os.File
		var data_analysis data_analysis_res

		json_file, err = os.Open(json_pth)
		if err != nil {
			log.Println("文件打开错误",err)
			return
		}
		defer func(json_file *os.File) {
			err := json_file.Close()
			if err != nil {
				return
			}
		}(json_file)
		dco := json.NewDecoder(json_file)
		err = dco.Decode(&data_analysis)
		if err != nil {
			log.Println("解析错误",err)
			return
		}
		//log.Println("data_res:",data_analysis.Action_analysis,data_analysis.Expression_analysis,data_analysis.Attention_analysis,data_analysis.Person_num)
		c.JSON(http.StatusOK,data_analysis)
	})

	err := r.Run(":43476")
	if err != nil {
		return 
	}
}

//func _main()  {
//	//r := utils.RedisClient
//	//r_list := r.LRange("done_Keys",0,-1).Val()
//	//for i:= range r_list{
//	//	println(i,r_list[i])
//	//}
//
//	//mg_cli := utils.MgoCli
//	//task_db := mg_cli.Database("task_db")
//	//info := task_db.Collection("info")
//	//println(info)
//}


