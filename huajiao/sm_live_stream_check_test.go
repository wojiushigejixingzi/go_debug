package huajiao

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"
)

func Test_sm_live_stream_check(t *testing.T) {
	check()
}

func check() {
	// 打开日志文件
	file, err := os.Open("/Users/wwb/Desktop/sm_live_image.txt")
	if err != nil {
		fmt.Println("无法打开文件:", err)
		return
	}
	defer file.Close()

	// 定义用于匹配 liveid 的正则表达式
	regex := regexp.MustCompile(`liveid:(\d+)`)

	// 创建一个切片用于存储所有的 liveid
	var liveIDs []string

	// 逐行扫描文件内容
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		// 查找并提取 liveid
		matches := regex.FindStringSubmatch(line)
		if len(matches) > 1 {
			liveIDs = append(liveIDs, matches[1])
		}
	}

	// 输出以逗号分隔的 liveid 列表
	fmt.Println(strings.Join(liveIDs, ","))

	// 错误处理
	if err := scanner.Err(); err != nil {
		fmt.Println("读取文件时出错:", err)
	}
}

type LogEntrys struct {
	Data struct {
		UserID int64 `json:"userid"`
	} `json:"data"`
}

func Test_exportUserId(t *testing.T) {
	exportUserId()
}

func exportUserId() {
	// 打开文件
	file, err := os.Open("/Users/wwb/Desktop/linkConnect.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()
	allUserId := []int64{}
	mapUserId := make(map[int64]int)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var entry LogEntrys
		err := json.Unmarshal([]byte(scanner.Text()), &entry)
		if err != nil {
			fmt.Println("Error parsing JSON:", err)
			continue
		}
		allUserId = append(allUserId, entry.Data.UserID)
		mapUserId[entry.Data.UserID]++
		// 打印 userid
		//fmt.Println("UserID:", entry.Data.UserID)
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
	}
	fmt.Println("CountAllUserId:", len(allUserId))
	fmt.Println("CountMapUserId:", len(mapUserId))

	fmt.Println("allUserId:", allUserId)
	fmt.Println("mapUserId:", mapUserId)

}
