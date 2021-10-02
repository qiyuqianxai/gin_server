package main

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"crypto/tls"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	_ "github.com/mattn/go-sqlite3"
)

func listDir(path, exclude string) []string {
	files, err := ioutil.ReadDir(path)
	var names []string
	if err != nil {
		log.Fatal(err)
		return names
	}

	for _, f := range files {
		if f.Name() != exclude {
			names = append(names, f.Name())
		}
	}
	return names
}

type Pair struct {
	Key   int
	Value int
}

type PairList []Pair

func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value > p[j].Value }

func sortMapByValue(m map[int]int) PairList {
	p := make(PairList, len(m))
	i := 0
	for k, v := range m {
		if k != -1 {
			p[i] = Pair{k, v}
			i = i + 1
		}
	}
	sort.Sort(p)
	return p
}

type dataFrame struct {
	rstToLn      map[int][]int
	rstToLabel   map[int]int
	labelToRst   map[int]int
	rstCount     map[int]int
	currentLabel int
	md5Hash      []byte
	lMax         int
	iMode        string
	lnToSid      map[int]string
	lnToMeta     map[int]string
	lnToInfo     map[int][]string
	sidToRst     map[string]int
	order        string
}

func computeMd5(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	defer file.Close()
	h := md5.New()
	if _, err := io.Copy(h, file); err != nil {
		log.Fatal(err)
		return nil, err
	}
	return h.Sum(nil), nil
}

func newDataFrame(path string, body bool, orderBySize bool) (*dataFrame, error) {
	rstCount := make(map[int]int)
	rstToLabel := make(map[int]int)
	labelToRst := make(map[int]int)
	rstToLn := make(map[int][]int)
	lnToSid := make(map[int]string)
	lnToSid[0] = "/workspace/www/static/no-image.jpg"
	lnToMeta := make(map[int]string)
	lnToInfo := make(map[int][]string)
	sidToRst := make(map[string]int)
	order := "count"
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	defer file.Close()
	h := md5.New()
	if _, err := io.Copy(h, file); err != nil {
		log.Fatal(err)
		return nil, err
	}
	md5Hash := h.Sum(nil)

	file.Seek(0, 0)
	scanner := bufio.NewScanner(file)
	ln := 1
	for scanner.Scan() {
		record := scanner.Text()
		ss := strings.Split(strings.TrimSpace(record), " ")
		if len(ss) >= 2 {
			sid, rst := ss[0], ss[1]
			if strings.Contains(sid, "\"") {
				sid = strings.ReplaceAll(sid, "\"", "")
			}
			lnToSid[ln] = sid
			if rst, err := strconv.Atoi(rst); err == nil {
				// if rst < -1 {
				// 	return nil, errors.New("数据集含有小于 -1 的rst")
				// }
				if body {
					sidToRst[sid] = rst
				}
				lns, ok := rstToLn[rst]
				if ok {
					lns = append(lns, ln)
					rstToLn[rst] = lns
				} else {
					rstToLn[rst] = []int{ln}
				}
				if len(ss) >= 3 {
					if _, err := strconv.Atoi(ss[2]); err != nil {
						return nil, fmt.Errorf("颜色框所在列需为数字，但是发现: %s", ss[2])
					}
					lnToMeta[ln] = ss[2]
					if len(ss) > 3 {
						lnToInfo[ln] = ss[3:]
					}
				}
			} else {
				return nil, fmt.Errorf("存在非法的label: %s", record)
			}
		} else {
			return nil, fmt.Errorf("存在信息不完整的记录: %s", record)
		}
		ln = ln + 1
	}
	_, ok := rstToLn[-1]
	if !ok {
		rstToLn[-1] = []int{0}
	}
	for k, v := range rstToLn {
		rstCount[k] = len(v)
	}
	lMax := -1
	if orderBySize {
		sortedRstByCount := sortMapByValue(rstCount)
		for i, p := range sortedRstByCount {
			rstToLabel[p.Key] = i + 1
			labelToRst[i+1] = p.Key
			if lMax < i+1 {
				lMax = i + 1
			}
		}
		rstToLabel[-1] = 0
		labelToRst[0] = -1
	} else {
		rsts := make([]int, 0, len(rstCount))
		for k := range rstCount {
			rsts = append(rsts, k)
		}
		sort.Ints(rsts)
		lMax = len(rsts)
		for i, k := range rsts {
			rstToLabel[k] = i
			labelToRst[i] = k
		}
		order = "label"
	}
	df := dataFrame{rstToLn: rstToLn, rstToLabel: rstToLabel,
		lnToSid: lnToSid, lnToMeta: lnToMeta, lnToInfo: lnToInfo,
		labelToRst: labelToRst, lMax: lMax, iMode: "hybrid",
		rstCount: rstCount, currentLabel: 1, md5Hash: md5Hash,
		sidToRst: sidToRst, order: order}
	return &df, nil
}

var ds map[string]*dataFrame
var serverAddr string
var rocStruct map[string]*RocFile

func getDf(path, body, order string) (*dataFrame, error) {
	df, err := newDataFrame(path, body == "on", order == "count")
	if err != nil {
		return nil, err
	}
	return df, nil
}

type bodyFrame struct {
	sid        []string
	itype      []int
	cameraID   []int
	trackID    []int
	timestamp  []int
	tsBeg      []int
	tsEnd      []int
	sRect      []string
	orgSid     []string
	cameraName []string
	score      []float32
	iColor     []int
	sidToColor map[string]int
	nColor     int
}

func loadDB(dbPath, dbname string) (string, error) {
	db, err := sql.Open("sqlite3", path.Join(dbPath, "BaseDataMap.db"))
	if err != nil {
		log.Fatal(err)
		return "", err
	}
	defer db.Close()
	q := `SELECT DISTINCT dbname FROM sid2db`
	rows, err := db.Query(q)
	if err != nil {
		log.Fatal(err)
		return "", err
	}
	defer rows.Close()
	uniDB := map[string]bool{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			log.Fatal(err)
			return "", err
		}
		uniDB[name+".db"] = true
	}
	if _, ok := uniDB[dbname]; ok {
		return "warning: 数据库已存在此数据", nil
	} else {
		q := `SELECT DISTINCT dbname FROM info`
		newdb, err := sql.Open("sqlite3", path.Join(dbPath, dbname))
		if err != nil {
			log.Fatal(err)
			return "打开数据库错误", err
		}
		defer newdb.Close()
		rows, err := newdb.Query(q)
		if err != nil {
			log.Fatal(err)
			return "数据库存在错误", err
		}
		defer rows.Close()
		for rows.Next() {
			var dbn string
			if err := rows.Scan(&dbn); err != nil {
				log.Fatal(err)
				return "", err
			}
			if dbn != dbname[:len(dbname)-3] {
				return "数据库名和 dbname 不一致", nil
			}
		}
		_, err2 := exec.Command("/bin/sh", "/workspace/load.sh", dbname).Output()
		if err2 != nil {
			log.Fatal(err2)
			return "导入数据错误", nil
		}
		return "导入数据成功", nil
	}
}

func chunkSlice(slice []string, chunkSize int) [][]string {
	var chunks [][]string
	for i := 0; i < len(slice); i += chunkSize {
		end := i + chunkSize

		// necessary check to avoid slicing beyond
		// slice capacity
		if end > len(slice) {
			end = len(slice)
		}

		chunks = append(chunks, slice[i:end])
	}

	return chunks
}

func queryBody(dbPath string, sids []string) (*bodyFrame, error) {
	dbToSid := make(map[string][]string)
	var unfoundSid []string
	cidTidToSid := make(map[int]map[int][]string)
	sidToColor := make(map[string]int)
	db, err := sql.Open("sqlite3", path.Join(dbPath, "BaseDataMap.db"))
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	defer db.Close()
	for _, s := range sids {
		stmt, err := db.Prepare("SELECT dbname FROM sid2db WHERE sid = ?")
		if err != nil {
			log.Fatal(err)
			return nil, err
		}
		defer stmt.Close()
		var name string
		err = stmt.QueryRow(s).Scan(&name)
		if err != nil {
			// log.Fatal(err)
			unfoundSid = append(unfoundSid, s)
			continue
		}

		if ss, ok := dbToSid[name]; ok {
			ss = append(ss, s)
			dbToSid[name] = ss
		} else {
			sss := []string{s}
			dbToSid[name] = sss
		}
	}

	itype, cameraID, trackID, timestamp, tsBeg, tsEnd :=
		map[string]int{}, map[string]int{}, map[string]int{},
		map[string]int{}, map[string]int{}, map[string]int{}
	sRect, cameraName, orgSid :=
		map[string]string{}, map[string]string{}, map[string]string{}
	score := map[string]float32{}

	for dbname, ssid := range dbToSid {
		bodyDb, err := sql.Open("sqlite3", path.Join(dbPath, dbname+".db"))
		if err != nil {
			log.Fatal(err)
			return nil, err
		}
		defer bodyDb.Close()
		chunkSize := 10000
		chunks := chunkSlice(ssid, chunkSize)
		for _, chunk := range chunks {
			q := `
				SELECT sid,itype,camera_id,track_id,timestamp,ts_beg,ts_end,snap_rect,org_sid,camera_name,score 
				FROM info 
				WHERE sid IN (?` + strings.Repeat(",?", len(chunk)-1) + `);`
			args := []interface{}{}
			for _, sid := range chunk {
				args = append(args, sid)
			}
			rows, err := bodyDb.Query(q, args...)
			if err != nil {
				log.Fatal(err)
				return nil, err
			}
			defer rows.Close()
			for rows.Next() {
				var itp, cid, tid, ttp, tsb, tse int
				var rect, cnm string
				var scr float32
				var sid, org string
				err = rows.Scan(&sid, &itp, &cid, &tid, &ttp, &tsb, &tse, &rect, &org, &cnm, &scr)
				if err != nil {
					log.Fatal(err)
					return nil, err
				}
				if itp == -1 {
					sidToColor[sid] = 0
				}
				if tidToSid, ok := cidTidToSid[cid]; ok {
					if tmpS, ok := tidToSid[tid]; ok {
						tmpS = append(tmpS, sid)
						tidToSid[tid] = tmpS
						cidTidToSid[cid] = tidToSid
					} else {
						ts := []string{sid}
						tidToSid[tid] = ts
						cidTidToSid[cid] = tidToSid
					}
				} else {
					tidToSid := make(map[int][]string)
					tidToSid[tid] = []string{sid}
					cidTidToSid[cid] = tidToSid
				}
				itype[sid] = itp
				cameraID[sid] = cid
				trackID[sid] = tid
				timestamp[sid] = ttp
				tsBeg[sid] = tsb
				tsEnd[sid] = tse
				sRect[sid] = rect
				orgSid[sid] = org
				cameraName[sid] = cnm
				score[sid] = scr
			}
		}
	}
	var i int = 1
	var itypeArr, cameraIDArr, trackIDArr, timestampArr, tsBegArr, tsEndArr, iColor []int
	var sRectArr, cameraNameArr, orgSidArr, cSidArr []string
	var scoreArr []float32
	for _, tidToSid := range cidTidToSid {
		for _, sids := range tidToSid {
			for _, sid := range sids {
				cSidArr = append(cSidArr, sid)
				itypeArr = append(itypeArr, itype[sid])
				cameraIDArr = append(cameraIDArr, cameraID[sid])
				trackIDArr = append(trackIDArr, trackID[sid])
				timestampArr = append(timestampArr, timestamp[sid])
				tsBegArr = append(tsBegArr, tsBeg[sid])
				tsEndArr = append(tsEndArr, tsEnd[sid])
				sRectArr = append(sRectArr, sRect[sid])
				orgSidArr = append(orgSidArr, orgSid[sid])
				cameraNameArr = append(cameraNameArr, cameraName[sid])
				scoreArr = append(scoreArr, score[sid])
				iColor = append(iColor, i)
			}
			i++
		}
	}
	var nColor int = i
	for _, sid := range unfoundSid {
		cSidArr = append(cSidArr, sid)
		itypeArr = append(itypeArr, -1)
		cameraIDArr = append(cameraIDArr, -1)
		trackIDArr = append(trackIDArr, -1)
		timestampArr = append(timestampArr, -1)
		tsBegArr = append(tsBegArr, -1)
		tsEndArr = append(tsEndArr, -1)
		sRectArr = append(sRectArr, "0,0,0,0")
		orgSidArr = append(orgSidArr, "")
		cameraNameArr = append(cameraNameArr, "")
		scoreArr = append(scoreArr, 0)
		iColor = append(iColor, 0)
	}
	return &bodyFrame{
		sid: cSidArr, itype: itypeArr, nColor: nColor, iColor: iColor,
		cameraID: cameraIDArr, trackID: trackIDArr, timestamp: timestampArr, tsBeg: tsBegArr,
		tsEnd: tsEndArr, sRect: sRectArr, orgSid: orgSidArr, cameraName: cameraNameArr,
		score: scoreArr}, nil
}

type BodyData struct {
	Storage string `form:"org"`
	Meta    string `form:"meta"`
	Rect    string `form:"rect"`
	Sid     string `form:"sid"`
	Camera  string `form:"camera"`
}

func bodyDetail(c *gin.Context) {
	var person BodyData
	if c.ShouldBindQuery(&person) == nil {
		image := "/starbox/starbox-prd-ai/" + person.Storage
		c.HTML(http.StatusOK, "body_detail.tmpl", gin.H{
			"rect":   person.Rect,
			"meta":   person.Meta,
			"sid":    person.Sid,
			"camera": person.Camera,
			"image":  image,
		})
	} else {
		c.HTML(http.StatusOK, "error.tmpl", gin.H{
			"server":  serverAddr,
			"message": "body 数据错误.",
		})
	}
}

type RstData struct {
	Name string `form:"dataset"`
	Sid  string `form:"sid"`
}

type RocFile struct {
	Roc RocData
	Md5 []byte
}

type RocData struct {
	FalsePairs map[string]ComPage `json:"false_pairs"`
	MissTops   MissTop1Pairs      `json:"miss_top1_pairs"`
	FilterPos  FilterPair         `json:"filtered_pos"`
	FilterNeg  FilterPair         `json:"filtered_neg"`
}

type ComPage struct {
	HighNegPairs []ComPair `json:"high_neg_pairs"`
	LowPosPairs  []ComPair `json:"low_pos_pairs"`
	Fpr          string    `json:"fpr"`
	Threshold    string    `json:"threshold"`
	Fnr          string    `json:"fnr"`
	XcName       string    `json:"xc_name"`
	HdName       string    `json:"hd_name"`
}

type MissTop1Pairs struct {
	Pairs  []ComPair `json:"pairs"`
	XcName string    `json:"xc_name"`
	HdName string    `json:"hd_name"`
	ExName string    `json:"ex_name"`
}
type ComPair struct {
	Xc       string `json:"xc"`
	Hd       string `json:"hd"`
	Score    string `json:"score"`
	Label    string `json:"label"`
	TopHd    string `json:"top_hd"`
	TopScore string `json:"top_score"`
	TestName string `json:"test_name"`
}

type FilterPair struct {
	Pairs  []ComPair `json:"pairs"`
	XcName string    `json:"xc_name"`
	HdName string    `json:"hd_name"`
}

func parseRoc(file string) (*RocFile, error) {
	jsonFile, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)
	var roc RocData
	json.Unmarshal(byteValue, &roc)
	md5Hash, err := computeMd5(file)
	if err != nil {
		return nil, err
	} else {
		rocFile := RocFile{roc, md5Hash}
		return &rocFile, nil
	}
}

func searchSid(c *gin.Context) {
	var sq RstData
	if c.ShouldBindQuery(&sq) == nil {
		if df, ok := ds[sq.Name]; ok {
			if rst, ok := df.sidToRst[sq.Sid]; ok {
				c.JSON(http.StatusOK, gin.H{
					"rst": rst,
				})
			} else {
				c.JSON(http.StatusOK, gin.H{"rst": "查无结果"})
			}
		} else {
			c.JSON(http.StatusOK, gin.H{"rst": "查无结果"})
		}
	} else {
		c.JSON(http.StatusBadRequest, gin.H{})
	}
}

func main() {
	rootPath := flag.String("root", "/ssd", "root folder")
	server := flag.String("server", "http://10.128.128.88:88", "server address")
	fileMode := flag.String("file", "false", "optional face file")
	flag.Parse()
	facePath := path.Join(*rootPath, "cluster_rst")
	bodyPath := path.Join(*rootPath, "body_rst")
	rocPath := path.Join(*rootPath, "roc_rst")
	dbPath := path.Join(*rootPath, "db")
	imgsDir := path.Join(*rootPath, "imgs")
	ds = make(map[string]*dataFrame)
	rocStruct = make(map[string]*RocFile)
	serverAddr = *server
	// fmt.Printf("root is %s\n", *rootPath)
	if *fileMode == "empty" {
		_ = os.Mkdir(facePath, 0755)
		_ = os.Mkdir(bodyPath, 0755)
		_ = os.Mkdir(dbPath, 0755)
		_ = os.Mkdir(imgsDir, 0755)
		_ = os.Mkdir(rocPath, 0755)
	}
	// } else {
	// 	resp, err := http.PostForm("/login",
	// 		url.Values{"file": {*faceFile}, "face": {""}, "body": {""}})
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	defer resp.Body.Close()
	// }

	r := gin.Default()
	r.LoadHTMLGlob("templates/*")

	r.GET("/", func(c *gin.Context) {
		faces := listDir(facePath, "_")
		bodies := listDir(bodyPath, "_")
		rocs := listDir(rocPath, "_")
		c.HTML(http.StatusOK, "index.tmpl", gin.H{
			"server": *server,
			"face":   faces,
			"body":   bodies,
			"roc":    rocs,
		})
	})

	r.Static("/static/imgs", imgsDir)
	r.NoRoute(func(c *gin.Context) {
		c.HTML(http.StatusOK, "error.tmpl", gin.H{
			"server":  *server,
			"message": "404: 地址错误",
		})
	})

	r.POST("/login", func(c *gin.Context) {
		face := c.PostForm("face")
		body := c.PostForm("body")
		roc := c.PostForm("roc")
		file := c.PostForm("file")
		order := c.PostForm("order")
		var name, root, isBody string
		if file == "" {
			if roc != "" {
				c.Redirect(http.StatusFound, fmt.Sprintf("/roc/%s", roc))
				return
			} else if face != "" {
				root = facePath
				name = face
				isBody = "off"
			} else {
				root = bodyPath
				name = body
				isBody = "on"
			}
			file = path.Join(root, name)
		} else {
			name = filepath.Base(file)
			isBody = "off"
		}

		if df, ok := ds[name]; !ok {
			dff, err := getDf(file, isBody, order)
			if err != nil {
				c.HTML(http.StatusOK, "error.tmpl", gin.H{
					"server":  *server,
					"message": err,
				})
			} else {
				ds[name] = dff
				c.Redirect(http.StatusFound, fmt.Sprintf("/details/%s/%s/%d", name, isBody, ds[name].currentLabel))
			}
		} else {
			md5Hash, err := computeMd5(file)
			if err != nil {
				c.HTML(http.StatusOK, "error.tmpl", gin.H{
					"server":  *server,
					"message": "计算md5错误.",
				})
			} else {
				if !bytes.Equal(md5Hash, df.md5Hash) || df.order != order {
					dff, err := getDf(file, isBody, order)
					if err != nil {
						c.HTML(http.StatusOK, "error.tmpl", gin.H{
							"server":  *server,
							"message": err,
						})
					} else {
						ds[name] = dff
						c.Redirect(http.StatusFound, fmt.Sprintf("/details/%s/%s/%d", name, isBody, ds[name].currentLabel))
					}
				} else {
					c.Redirect(http.StatusFound, fmt.Sprintf("/details/%s/%s/%d", name, isBody, ds[name].currentLabel))
				}
			}
		}
	})

	r.GET("/roc/:name", func(c *gin.Context) {
		name := c.Param("name")
		file := path.Join(rocPath, name)
		if _, ok := rocStruct[name]; !ok {
			roc, err := parseRoc(file)
			if err != nil {
				c.HTML(http.StatusOK, "error.tmpl", gin.H{
					"server":  *server,
					"message": err,
				})
			} else {
				rocStruct[name] = roc
			}
		} else {
			md5Hash, err := computeMd5(file)
			if err != nil {
				c.HTML(http.StatusOK, "error.tmpl", gin.H{
					"server":  *server,
					"message": "计算md5错误.",
				})
			} else {
				if !bytes.Equal(md5Hash, rocStruct[name].Md5) {
					roc, err := parseRoc(file)
					if err != nil {
						c.HTML(http.StatusOK, "error.tmpl", gin.H{
							"server":  *server,
							"message": err,
						})
					} else {
						rocStruct[name] = roc
					}
				}
			}
		}
		roc := rocStruct[name]
		fpr := []string{"missed_top1_pairs", "filtered"}
		for _, page := range roc.Roc.FalsePairs {
			fpr = append(fpr, fmt.Sprintf("fpr=%s", page.Fpr))
		}
		c.HTML(http.StatusOK, "roc.tmpl", gin.H{
			"name":    "fr-test",
			"dataset": name,
			"fpr":     fpr,
			"page":    roc.Roc.FalsePairs,
		})
	})

	r.GET("/roc-data/:name/:fpr/:ftype/:loc", func(c *gin.Context) {
		name := c.Param("name")
		fpr := c.Param("fpr")
		ftype := c.Param("ftype")
		loc := c.Param("loc")
		sendSize := 20
		if roc, ok := rocStruct[name]; !ok {
			c.HTML(http.StatusOK, "error.tmpl", gin.H{
				"server":  *server,
				"message": fmt.Sprintf("%s 数据尚未载入.", name),
			})
		} else {
			if loc, err := strconv.Atoi(loc); err == nil {
				hd := []string{}
				xc := []string{}
				score := []string{}
				label := []string{}
				testName := []string{}
				topHd := []string{}
				topScore := []string{}
				var pairs []ComPair
				var hdName string
				var xcName string
				exName := "nan"
				if ftype == "miss" {
					pairs = roc.Roc.MissTops.Pairs
					hdName = roc.Roc.MissTops.HdName
					xcName = roc.Roc.MissTops.XcName
					exName = roc.Roc.MissTops.ExName
				} else if ftype == "filtered-pos" {
					pairs = roc.Roc.FilterPos.Pairs
					hdName = roc.Roc.FilterPos.HdName
					xcName = roc.Roc.FilterPos.XcName
				} else if ftype == "filtered-neg" {
					pairs = roc.Roc.FilterNeg.Pairs
					hdName = roc.Roc.FilterNeg.HdName
					xcName = roc.Roc.FilterNeg.XcName
				} else {
					page := roc.Roc.FalsePairs[fpr]
					hdName = page.HdName
					xcName = page.XcName
					if ftype == "high-neg" {
						pairs = page.HighNegPairs
					} else {
						pairs = page.LowPosPairs
					}
				}
				if loc < len(pairs) {
					bound := loc + sendSize
					if bound > len(pairs) {
						bound = len(pairs)
					}
					for _, p := range pairs[loc:bound] {
						hd = append(hd, p.Hd)
						xc = append(xc, p.Xc)
						score = append(score, p.Score)
						label = append(label, p.Label)
						testName = append(testName, p.TestName)
						if ftype == "miss" {
							topScore = append(topScore, p.TopScore)
							topHd = append(topHd, p.TopHd)
						}
					}
				}
				c.JSON(http.StatusOK, gin.H{
					"hd":        hd,
					"xc":        xc,
					"score":     score,
					"label":     label,
					"top_hd":    topHd,
					"top_score": topScore,
					"hd_name":   hdName,
					"xc_name":   xcName,
					"ex_name":   exName,
					"test_name": testName,
				})
			} else {
				c.HTML(http.StatusOK, "error.tmpl", gin.H{
					"server":  *server,
					"message": fmt.Sprintf("loc 需为有效数字."),
				})
			}
		}
	})

	r.GET("/roc-size/:name/:fpr/:ftype", func(c *gin.Context) {
		name := c.Param("name")
		fpr := c.Param("fpr")
		ftype := c.Param("ftype")
		if roc, ok := rocStruct[name]; !ok {
			c.HTML(http.StatusOK, "error.tmpl", gin.H{
				"server":  *server,
				"message": fmt.Sprintf("%s 数据尚未载入.", name),
			})
		} else {
			var pairs []ComPair
			if ftype == "miss" {
				pairs = roc.Roc.MissTops.Pairs
			} else if ftype == "filtered-pos" {
				pairs = roc.Roc.FilterPos.Pairs
			} else if ftype == "filtered-neg" {
				pairs = roc.Roc.FilterNeg.Pairs
			} else {
				page := roc.Roc.FalsePairs[fpr]
				if ftype == "high-neg" {
					pairs = page.HighNegPairs
				} else {
					pairs = page.LowPosPairs
				}
			}
			c.JSON(http.StatusOK, gin.H{
				"size": len(pairs),
			})
		}
	})

	r.GET("/details/:name/:body/:loc", func(c *gin.Context) {
		name := c.Param("name")
		body := c.Param("body")
		loc := c.Param("loc")
		if loc, err := strconv.Atoi(loc); err == nil {
			if df, ok := ds[name]; ok {
				lMax := df.lMax
				if loc >= 0 && loc <= lMax {
					df.currentLabel = loc
					if body == "on" {
						c.HTML(http.StatusOK, "review_body.tmpl", gin.H{
							"name":    "stars",
							"short":   name,
							"dataset": name,
							"current": loc,
							"lmax":    lMax,
							"imode":   df.iMode,
						})
					} else {
						c.HTML(http.StatusOK, "review_face.tmpl", gin.H{
							"name":    "stars",
							"short":   name,
							"dataset": name,
							"current": loc,
							"lmax":    lMax - 1,
						})
					}
				} else {
					c.HTML(http.StatusOK, "error.tmpl", gin.H{
						"server":  *server,
						"message": "label 超出范围.",
					})
				}
			} else {
				c.HTML(http.StatusOK, "error.tmpl", gin.H{
					"server":  *server,
					"message": fmt.Sprintf("%s 数据集尚未载入.", name),
				})
			}
		} else {
			log.Fatal(err)
			c.HTML(http.StatusOK, "error.tmpl", gin.H{
				"server":  *server,
				"message": "label 需为有效数字.",
			})
		}
	})

	r.GET("/get_current_rst/:name/:loc", func(c *gin.Context) {
		name := c.Param("name")
		loc := c.Param("loc")
		if loc, err := strconv.Atoi(loc); err == nil {
			if df, ok := ds[name]; ok {
				currentRst := df.labelToRst[loc]
				count := df.rstCount[currentRst]
				c.JSON(http.StatusOK, gin.H{
					"rst":  currentRst,
					"size": count,
				})
			} else {
				c.JSON(http.StatusBadRequest, gin.H{})
			}
		} else {
			c.JSON(http.StatusBadRequest, gin.H{})
		}
	})

	r.GET("/rst_to_label/:name/:rst", func(c *gin.Context) {
		name := c.Param("name")
		rst := c.Param("rst")
		if rst, err := strconv.Atoi(rst); err == nil {
			if df, ok := ds[name]; ok {
				label := df.rstToLabel[rst]
				c.JSON(http.StatusOK, gin.H{
					"label": label,
				})
			} else {
				c.JSON(http.StatusBadRequest, gin.H{})
			}
		} else {
			c.JSON(http.StatusBadRequest, gin.H{})
		}
	})

	r.GET("/get_index_labels/:name", func(c *gin.Context) {
		name := c.Param("name")
		if df, ok := ds[name]; ok {
			labelToRst := df.labelToRst
			c.JSON(http.StatusOK, gin.H{
				"index_to_label": labelToRst,
			})
		} else {
			c.HTML(http.StatusOK, "error.tmpl", gin.H{
				"server":  *server,
				"message": fmt.Sprintf("%s 数据集尚未载入.", name),
			})
		}
	})

	r.GET("/templates/:size/:name/:loc", func(c *gin.Context) {
		name := c.Param("name")
		// size := c.Param("size")
		loc := c.Param("loc")
		if loc, err := strconv.Atoi(loc); err == nil {
			if df, ok := ds[name]; ok {
				lns := df.rstToLn[loc]
				sids := []string{}
				meta := []string{}
				info := map[int][]string{}
				for _, s := range lns {
					sids = append(sids, df.lnToSid[s])
				}
				if len(df.lnToMeta) > 0 {
					for _, s := range lns {
						meta = append(meta, df.lnToMeta[s])
					}
				}
				if len(df.lnToInfo) > 0 {
					for i, s := range lns {
						info[i] = df.lnToInfo[s]
					}
				}
				c.JSON(http.StatusOK, gin.H{
					"imgpath": sids,
					"meta":    meta,
					"info":    info,
				})
			} else {
				c.JSON(http.StatusBadRequest, gin.H{
					"error": err.Error(),
				})
			}
		} else {
			c.JSON(http.StatusBadRequest, gin.H{})
		}
	})

	r.GET("/update_label/:size/:name/:loc", func(c *gin.Context) {
		name := c.Param("name")
		loc := c.Param("loc")
		if loc, err := strconv.Atoi(loc); err == nil {
			if df, ok := ds[name]; ok {
				df.currentLabel = loc
				c.String(http.StatusOK, "success")
			} else {
				c.String(http.StatusBadRequest, "error")
			}
		} else {
			c.String(http.StatusBadRequest, "error")
		}
	})

	r.GET("/update_imode/:name/:imode", func(c *gin.Context) {
		name := c.Param("name")
		imode := c.Param("imode")

		if df, ok := ds[name]; ok {
			df.iMode = imode
			c.String(http.StatusOK, "success")
		} else {
			c.String(http.StatusBadRequest, "error")
		}

	})

	r.GET("/templates-body/:size/:name/:loc", func(c *gin.Context) {
		name := c.Param("name")
		// size := c.Param("size")
		loc := c.Param("loc")
		if loc, err := strconv.Atoi(loc); err == nil {
			if df, ok := ds[name]; ok {
				lns := df.rstToLn[loc]
				sids := []string{}
				for _, s := range lns {
					sids = append(sids, df.lnToSid[s])
				}
				bf, err := queryBody(dbPath, sids)
				if err == nil {
					c.JSON(http.StatusOK, gin.H{
						"imgpath": bf.sid, "itype": bf.itype, "camera_id": bf.cameraID,
						"track_id": bf.trackID, "timestamp": bf.timestamp,
						"ts_beg": bf.tsBeg, "ts_end": bf.tsEnd, "rects": bf.sRect,
						"org_sid": bf.orgSid, "camera_name": bf.cameraName, "score": bf.score,
						"icolor": bf.iColor, "ncolor": bf.nColor,
					})
				} else {
					c.JSON(http.StatusBadRequest, gin.H{})
				}
			} else {
				c.JSON(http.StatusBadRequest, gin.H{})
			}
		} else {
			c.JSON(http.StatusBadRequest, gin.H{})
		}
	})

	r.GET("/starbox-raw/:storage", func(c *gin.Context) {
		storage := c.Param("storage")
		url := "http://ks3.kylin.cloudwalk.work/starbox-prd-ai/" + storage
		response, err := http.Get(url)
		if err != nil || response.StatusCode != http.StatusOK {
			c.Status(http.StatusServiceUnavailable)
			return
		}

		reader := response.Body
		contentLength := response.ContentLength
		// contentType := response.Header.Get("Content-Type")

		extraHeaders := map[string]string{
			// "Content-Disposition": `attachment; filename="go.png"`,
		}

		c.DataFromReader(http.StatusOK, contentLength, "image/jpg", reader, extraHeaders)
	})

	r.GET("/performance-raw/:storage", func(c *gin.Context) {
		storage := c.Param("storage")
		url := fmt.Sprintf("http://starbox.cloudwalk.work/performance/file/download?storageId=%s&type=1&filename=tmp.jpg", storage)
		response, err := http.Get(url)
		if err != nil || response.StatusCode != http.StatusOK {
			c.Status(http.StatusServiceUnavailable)
			return
		}

		reader := response.Body
		contentLength := response.ContentLength
		contentType := response.Header.Get("Content-Type")

		extraHeaders := map[string]string{
			// "Content-Disposition": `attachment; filename="go.png"`,
		}

		c.DataFromReader(http.StatusOK, contentLength, contentType, reader, extraHeaders)
	})

	r.GET("/image-query", func(c *gin.Context) {
		pool := c.Query("pool")
		id := c.Query("id")
		server := c.Query("server")
		url := fmt.Sprintf("https://%s?pool=%s&id=%s", server, pool, id)
		client := &http.Client{
			Transport: &http.Transport{
				Proxy: http.ProxyFromEnvironment,
				DialContext: (&net.Dialer{
					Timeout:   10 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,

				MaxConnsPerHost:     96,
				MaxIdleConns:        0,
				MaxIdleConnsPerHost: 96,
				IdleConnTimeout:     time.Duration(15) * time.Second,
				TLSClientConfig:     &tls.Config{InsecureSkipVerify: true}, //跳过证书验证
			},
			Timeout: 20 * time.Second,
		}
		response, err := client.Get(url)
		if err != nil {
			c.Status(http.StatusServiceUnavailable)
			return
		}

		reader := response.Body
		defer reader.Close()
		contentLength := response.ContentLength
		contentType := response.Header.Get("Content-Type")

		extraHeaders := map[string]string{
			"Content-Disposition": `inline;`,
		}

		c.DataFromReader(http.StatusOK, contentLength, contentType, reader, extraHeaders)
	})

	r.GET("/admin", func(c *gin.Context) {
		dbs := listDir(dbPath, "BaseDataMap.db")
		c.HTML(http.StatusOK, "admin.tmpl", gin.H{
			"server": *server,
			"db":     dbs,
		})
	})

	r.POST("/load-db", func(c *gin.Context) {
		db := c.PostForm("db")
		res, err := loadDB(dbPath, db)
		if err != nil {
			c.HTML(http.StatusOK, "error.tmpl", gin.H{
				"server":  *server,
				"message": err,
			})
		} else {
			c.String(http.StatusOK, res)
		}
	})

	r.Any("/search_sid", searchSid)
	r.Any("/starbox-body-raw", bodyDetail)
	r.Run(":8080")
}
