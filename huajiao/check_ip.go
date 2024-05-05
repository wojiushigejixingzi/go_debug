package huajiao

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/transform"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type Location struct {
	Country  string `json:"country"`
	Province string `json:"province"`
	City     string `json:"city"`
}

type LogEntry struct {
	Time    string
	Context struct {
		URI string `json:"uri"`
	} `json:"context"`
	Data struct {
		Location map[string]Location `json:"location"`
		IP       string              `json:"ip"`
	} `json:"data"`
}

type csvData struct {
	Ip              string
	UserId          string
	Address_huajiao string
	Address_sanfang string
	time            string
}

type ResponseData struct {
	Ip          string `json:"ip"`
	Pro         string `json:"pro"`
	ProCode     string `json:"proCode"`
	City        string `json:"city"`
	CityCode    string `json:"cityCode"`
	Region      string `json:"region"`
	RegionCode  string `json:"regionCode"`
	Addr        string `json:"addr"`
	RegionNames string `json:"regionNames"`
	Err         string `json:"err"`
}

func Check_ip() {
	csvD := []csvData{}
	// 打开文件
	file, err := os.Open("/Users/wwb/Desktop/ip_check.txt")
	if err != nil {
		fmt.Println("无法打开文件:", err)
		return
	}
	defer file.Close()
	// 创建一个 Scanner 用于逐行读取文件内容
	scanner := bufio.NewScanner(file)

	i := 1
	// 逐行读取文件内容并解析 JSON
	for scanner.Scan() {
		var csvTemp csvData
		var logEntry LogEntry
		var sanfangResponse ResponseData
		if err := json.Unmarshal(scanner.Bytes(), &logEntry); err != nil {
			fmt.Println("解析日志失败:", err)
			continue
		}

		// 打印每条日志中的 IP 数据
		fmt.Println("IP:", logEntry.Data.IP, i)
		i++

		// 发送 HTTP 请求
		resp, err := http.Get(fmt.Sprintf("http://whois.pconline.com.cn/ipJson.jsp?json=true&ip=%s", logEntry.Data.IP))
		if err != nil {
			fmt.Println("发送请求失败:", err)
			continue
		}
		defer resp.Body.Close()

		// 读取响应内容并输出
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			fmt.Println("读取响应失败:", err)
			continue
		}
		// 将 GB2312 编码转换为 UTF-8 编码
		decoder := simplifiedchinese.GB18030.NewDecoder()
		reader := transform.NewReader(bytes.NewReader(body), decoder)
		utf8Body, err := ioutil.ReadAll(reader)
		if err != nil {
			fmt.Println("转换编码失败:", err)
			continue
		}
		stringSangFang := string(utf8Body)
		if err := json.Unmarshal([]byte(stringSangFang), &sanfangResponse); err != nil {
			fmt.Println("解析日志失败:", err)
		}
		uriParts := strings.Split(logEntry.Context.URI, "?")
		if len(uriParts) > 1 {
			queryParts := strings.Split(uriParts[1], "&")
			for _, part := range queryParts {
				paramParts := strings.Split(part, "=")
				if len(paramParts) == 2 && paramParts[0] == "userid" {
					fmt.Println("UserID:", paramParts[1])
					csvTemp.UserId = paramParts[1]
					break
				}
			}
		} else {
			fmt.Println("URI 格式错误:", logEntry.Context.URI)
		}
		csvTemp.Ip = logEntry.Data.IP
		csvTemp.time = logEntry.Time
		csvTemp.Address_huajiao = logEntry.Data.Location[logEntry.Data.IP].Country + logEntry.Data.Location[logEntry.Data.IP].Province + logEntry.Data.Location[logEntry.Data.IP].City
		csvTemp.Address_sanfang = sanfangResponse.Addr
		// 输出响应内容
		fmt.Println("API 响应:", string(utf8Body))
		//休眠50ms
		time.Sleep(200 * time.Millisecond)
		csvD = append(csvD, csvTemp)
		//if i > 2 {
		//	break
		//}
	}
	fmt.Println("总数量", i)
	////生成csv
	csv_file, err := os.Create("/Users/wwb/Desktop/check_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer csv_file.Close()
	writer := csv.NewWriter(csv_file)
	defer writer.Flush()
	// 写入标题行
	err = writer.Write([]string{"IP", "userId", "花椒识别地址", "三方识别地址", "上麦拦截时间"})
	if err != nil {
		log.Fatal(err)
	}
	for _, data := range csvD {
		err = writer.Write([]string{data.Ip, data.UserId, data.Address_huajiao, data.Address_sanfang, data.time})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(data.Ip + ":" + data.Address_huajiao + ":" + data.Address_sanfang + ":写入成功")
	}

	// 写入数据行
	//for _,data := range csvD {
	//	err = writer.Write([]string{ip, loc.Country, loc.Province, loc.City,
	//		strconv.FormatBool(req.IsOverseaRequest),
	//		strconv.FormatBool(req.IsChineseLocation),
	//		strconv.FormatBool(req.IsForbiddenChineseProvince),
	//		req.ForwardIP, req.RemoteIP})
	//	if err != nil {
	//		log.Fatal(err)
	//	}
	//}
}
