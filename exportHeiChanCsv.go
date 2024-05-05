package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type LogHeiChanEntry struct {
	Time string      `json:"time"`
	Data HeiChanData `json:"data"`
}

//日志原始文件路径
var logFilePath = "/Users/wwb/Desktop/huajiao/文件/黑产数据统计/日志原始文件"

//日志导出的csv文件路径
var csvFilePath = "/Users/wwb/Desktop/huajiao/文件/黑产数据统计/日志导出的csv文件"

type HeiChanData struct {
	Sender    string `json:"sender"`
	UserLevel int    `json:"userLevel"`
	Content   string `json:"content"`
	Receiver  string `json:"receiver"`
	IP        string `json:"ip"`
	DeviceID  string `json:"deviceid"`
	SMID      string `json:"smid"`
	Nickname  string `json:"nickname"`
}

func EchoCsv() {
	//当天的年月日日期，格式如下：2023-11-9
	timeNow := time.Now().Format("2006-01-02")

	logFilePath := logFilePath + "/AOM-" + timeNow + ".txt"
	csvFilePath := csvFilePath + "/黑产指纹拦截统计_" + timeNow + ".csv"

	// 打开日志文件
	logFile, err := os.Open(logFilePath)
	if err != nil {
		fmt.Printf("无法打开日志文件：%v\n", err)
		return
	}
	defer logFile.Close()

	// 创建CSV文件
	csvFile, err := os.Create(csvFilePath)
	if err != nil {
		fmt.Printf("无法创建CSV文件：%v\n", err)
		return
	}
	defer csvFile.Close()

	// 创建CSV写入器
	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()

	// 写入CSV文件的标题行
	header := []string{"时间", "发送弹幕uid", "用户昵称", "用户等级", "弹幕内容", "直播id", "IP", "设备DeviceID", "SMID"}
	csvWriter.Write(header)

	// 逐行读取日志文件
	scanner := bufio.NewScanner(logFile)
	for scanner.Scan() {
		line := scanner.Text()
		var logEntry LogHeiChanEntry

		// 解析JSON数据
		err := json.Unmarshal([]byte(line), &logEntry)
		if err != nil {
			fmt.Printf("解析JSON失败：%v\n", err)
			continue
		}

		// 提取data字段的数据
		data := logEntry.Data

		//time格式为：2023-11-08 01:33:30.881180，对time进行截取只保留完整的年月日时分秒
		timess := logEntry.Time
		timess = timess[0:19]
		// 将提取的数据写入CSV文件
		record := []string{
			timess,
			data.Sender,
			data.Nickname,
			fmt.Sprintf("%d", data.UserLevel),
			data.Content,
			data.Receiver,
			data.IP,
			data.DeviceID,
			data.SMID,
		}
		csvWriter.Write(record)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("读取日志文件失败：%v\n", err)
		return
	}
}
