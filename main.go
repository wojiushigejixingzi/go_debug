package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	//"go_debug/huajiao"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"os"
	"sort"
	"time"

	"github.com/gin-gonic/gin"
)

func debu() {
	url := "http://dev.cpmain.cn:9010/live/room/linkApply/accept?userid=173182618&time=1704338267&rand=0.7663434271901584&platform=ios&version=1.0&guid=b2bbd979c644d839ce61933792bccb60"
	method := "POST"

	payload := &bytes.Buffer{}
	writer := multipart.NewWriter(payload)
	_ = writer.WriteField("linkid", "5308")
	err := writer.Close()
	if err != nil {
		fmt.Println(err)
		return
	}

	client := &http.Client{}
	req, err := http.NewRequest(method, url, payload)

	if err != nil {
		fmt.Println(err)
		return
	}
	req.Header.Add("User-Agent", "Apifox/1.0.0 (https://apifox.com)")
	req.Header.Add("Cookie", "token=YWEKUo6aZb2wMOFUDg--iPIREVurjOFH48e4;undefined")
	req.Header.Add("Accept", "*/*")
	req.Header.Add("Host", "dev.cpmain.cn:9010")
	req.Header.Add("Connection", "keep-alive")
	req.Header.Add("Content-Type", "multipart/form-data; boundary=--------------------------900622267622359747237815")

	req.Header.Set("Content-Type", writer.FormDataContentType())
	res, err := client.Do(req)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer res.Body.Close()

	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(body))
}

func main() {
	//huajiao.Check_ip()
	//tool.Export_barrage_feed()
	//循环调用debu两次
	//for i := 0; i < 2; i++ {
	//	debu()
	//}

	//HotTimeUse()
	//url := "http://image.huajiao.com/beef96add3d5a2133c311481ef8f3e63.jpg"
	//types := mime.TypeByExtension(url)
	//fmt.Println(types)
	//tool.ExampleClient()
	// 1.创建路由
	r := gin.Default()
	// 2.绑定路由规则，执行的函数
	// gin.Context，封装了request和response
	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "hello World!")
	})
	// 3.监听端口，默认在8080
	// Run("里面不指定端口号默认为8080")
	r.Run(":8011")
}

type ResponseData struct {
	Data []struct {
		URL string `json:"url"`
	} `json:"data"`
}

func longestConsecutive(nums []int) int {
	numHash := map[int]bool{}
	for _, num := range nums {
		numHash[num] = true
	}
	cnt := 0
	for v := range numHash {
		if numHash[v-1] {
			continue
		}
		temp := 1
		for numHash[v+1] {
			cnt++
			v++
		}
		cnt = max(cnt, temp)
	}
	return cnt
}
func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func getUrl() {
	// 获取当前时间戳
	currentTime := time.Now().Unix()
	// 计算 30 秒前的时间戳
	stime := currentTime - 30
	// 构建请求 URL
	//url := fmt.Sprintf("http://jiantugo.huajiao.com/api/getVedioImgList?etime=%d&liveid=341501889&stime=%d&type=LIVE&with_total=0", currentTime, stime)
	url := fmt.Sprintf("http://jiantugo.huajiao.com/api/getVedioImgList?etime=%d&liveid=341499416&stime=%d&type=LIVE&with_total=0", currentTime, stime)
	//打印 currentTime 和 stime
	fmt.Println("当前时间戳:", currentTime)
	fmt.Println("30秒前的时间戳:", stime)
	// 发起 GET 请求
	response, err := http.Get(url)
	if err != nil {
		fmt.Println("请求失败:", err)
		return
	}
	defer response.Body.Close()

	// 解析 JSON 响应
	var responseData ResponseData
	decoder := json.NewDecoder(response.Body)
	if err := decoder.Decode(&responseData); err != nil {
		fmt.Println("解析JSON失败:", err)
		return
	}

	// 打印 URL
	fmt.Println("URL列表:")
	for _, item := range responseData.Data {
		fmt.Println(item.URL)
	}
}

type LogEntry struct {
	Data struct {
		Info struct {
			MgID  string `json:"mg_id"`
			Score int    `json:"score"`
		} `json:"report_msg"`
	} `json:"data"`
}

type LogNoRecordEntry struct {
	Data struct {
		Info struct {
			MgID   string `json:"mg_id"`
			Report struct {
				Results []struct {
					Uid   string `json:"uid"`
					Award int    `json:"award"`
				} `json:"results"`
			} `json:"report_msg"`
		} `json:"data.info"`
	} `json:"data"`
}

//从本地读取文件,目录如下：/Users/wwb/Desktop/aom.txt
func readFile() {

	mgIdMapName := make(map[string]string)
	mgIdMapName["1468180338417074177"] = "飞行棋"
	mgIdMapName["1461227817776713818"] = "碰碰我最强"
	mgIdMapName["1461297734886621238"] = "五子棋"
	mgIdMapName["1472142640866779138"] = "排雷兵"
	mgIdMapName["1472142559912517633"] = "乌诺牌"

	///从本地读取文件,目录如下：/Users/wwb/Desktop/aom.txt
	file, err := os.Open("/Users/wwb/Desktop/aom.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	// 创建一个映射用于存储每个mgId的score总和
	scoreSumByMgID := make(map[string]int)
	for scanner.Scan() {
		// 解析每行日志
		var log LogEntry
		if err := json.Unmarshal([]byte(scanner.Text()), &log); err != nil {
			fmt.Println("解析日志失败:", err)
			continue
		}
		// 将score添加到对应的mgId的总和中
		scoreSumByMgID[log.Data.Info.MgID] += log.Data.Info.Score
	}
	if err := scanner.Err(); err != nil {
		fmt.Println("读取日志文件失败:", err)
		return
	}
	// 输出每个mgId的score总和
	for mgID, totalScore := range scoreSumByMgID {
		fmt.Printf("gameName:%s, Total Score: %d\n", mgIdMapName[mgID], totalScore)
	}
	//fmt.Println("检测未计入榜单数据")
	//return
	//file_no_record, err := os.Open("/Users/wwb/Desktop/aom_no_record.txt")
	//if err != nil {
	//	panic(err)
	//}
	//defer file_no_record.Close()
	//scoreSumNoRecordByMgID := make(map[string]int)
	//scanner_no_record := bufio.NewScanner(file_no_record)
	//for scanner_no_record.Scan() {
	//	// 解析每行日志
	//	var log LogNoRecordEntry
	//	if err := json.Unmarshal([]byte(scanner_no_record.Text()), &log); err != nil {
	//		fmt.Println("解析日志失败:", err)
	//		continue
	//	}
	//	// 遍历results中的award值并累加到对应的mgId
	//	for _, result := range log.Data.Info.Report.Results {
	//		if result.Uid != string(250230291) {
	//			continue
	//		}
	//		scoreSumNoRecordByMgID[log.Data.Info.MgID] += result.Award
	//	}
	//
	//}
	//for msgId, totalScore := range scoreSumNoRecordByMgID {
	//	fmt.Printf("gameName:%s, Total Score: %d\n", mgIdMapName[msgId], totalScore)
	//}

}

//字母异位词分组
func groupAnagrams(strs []string) [][]string {
	mp := map[string][]string{}
	for _, str := range strs {
		s := []byte(str)
		sort.Slice(s, func(i, j int) bool {
			return s[i] < s[j]
		})
		sortedStr := string(s)
		mp[sortedStr] = append(mp[sortedStr], str)
	}
	ans := make([][]string, 0, len(mp))
	for _, v := range mp {
		ans = append(ans, v)
	}
	return ans
}

//快速排序算法
func quickSort() []int {
	arr := []int{1, 2, 3, 4, 5, 6, 7, 8}
	return arr
}

func towSum(nums []int, target int) []int {
	hash := map[int]int{}
	for p, num := range nums {
		if q, ok := hash[target-num]; ok {
			return []int{p, q}
		}
		hash[num] = p
	}
	return nil
}
