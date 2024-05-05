package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type log_struct struct {
	TimeUse float64 `json:"time_use"`
	Name    string  `json:"name"`
	It      string  `json:"it"`
	TraceId string  `json:"trace_id"`
	Time    string  `json:"time"`
}

type LogData struct {
	Level   string `json:"level"`
	Time    string `json:"time"`
	App     string `json:"app"`
	TraceId string `json:"trace_id"`
	// ... 其他字段可以继续添加
	Context struct {
		It string `json:"it"`
	}
	Data struct {
		Result string `json:"result"`
	} `json:"data"`
	// ... 其他字段可以继续添加
}

func HotTimeUse() {
	// 读取日志文件
	fileContent, err := ioutil.ReadFile("/Users/wwb/Desktop/huajiao/文件/热门日志分析/2024-1-2-card-slow.txt")
	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	logs := strings.Split(string(fileContent), "\n")

	// 准备CSV文件写入器
	csvFile, err := os.Create("/Users/wwb/Desktop/huajiao/文件/热门日志分析/2024-1-2_log_features.csv")
	if err != nil {
		fmt.Println("Error creating CSV file:", err)
		return
	}
	defer csvFile.Close()

	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()

	// 写入CSV文件头部
	csvWriter.Write([]string{"log_id", "it", "time", "trace_id", "max_offset", "name"})

	// 遍历日志数据并处理
	for idx, logStr := range logs {
		var log LogData
		if err := json.Unmarshal([]byte(logStr), &log); err != nil {
			continue // 跳过解析错误的日志行
		}

		// 提取"result"字段中的偏移量数据
		logs := extractOffsets(log.Data.Result, log.Context.It, log.TraceId, log.Time)

		// 计算相邻偏移量的最大值
		max_logs := calculateMaxOffset(logs)

		// 将结果写入CSV文件
		csvWriter.Write([]string{strconv.Itoa(idx), max_logs.TraceId, max_logs.Time, max_logs.It, fmt.Sprintf("%.4f", max_logs.TimeUse), max_logs.Name})
	}

	fmt.Println("CSV file created successfully.")
}

func extractOffsets(result, it, traceId, time string) []log_struct {
	offsets := []float64{}
	re := regexp.MustCompile(`.*?offset: (.*?)[\n\r]`)
	ra := regexp.MustCompile(`.*?├─(.*?)[\n\r]`)
	matches := re.FindAllStringSubmatch(result, -1)
	matches_ra := ra.FindAllStringSubmatch(result, -1)
	//for _, matches_ra := range matches_ra {
	//	name := matches_ra[1]
	//	fmt.Println(name)
	//}
	res := []log_struct{}
	for index, match := range matches {
		var log log_struct
		offset, err := strconv.ParseFloat(match[1], 64)
		if err == nil {
			offsets = append(offsets, offset)
			log.TimeUse = offset
		}
		if index == 0 {
			continue
		}
		log.Name = matches_ra[index][1]
		log.It = it
		log.TraceId = traceId
		log.Time = time

		res = append(res, log)
	}

	return res
}

func calculateMaxOffset(logs []log_struct) log_struct {
	if len(logs) == 0 {
		return log_struct{}
	}

	max := logs[0].TimeUse
	name := ""
	for i := 1; i < len(logs); i++ {
		diff := logs[i].TimeUse - logs[i-1].TimeUse
		if diff > max {
			max = diff
			name = logs[i].Name
		}
	}
	log := log_struct{}
	log.Name = name
	log.TimeUse = max
	log.It = logs[0].It
	log.TraceId = logs[0].TraceId
	log.Time = logs[0].Time

	return log
}
