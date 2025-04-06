package huajiao

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"
)

// 定义JSON日志结构
type Log14Entry struct {
	Data struct {
		Message struct {
			GiftName      string `json:"gift_name"`
			TotalAmount   int    `json:"total_amount"`
			TotalDiscount int    `json:"total_discount"`
			TsID          int64  `json:"ts_id"`
			Receivers     []any  `json:"receivers"`
			Time          int64  `json:"time"`
		} `json:"message"`
	} `json:"data"`
}

func Test_export(t *testing.T) {
	// 打开日志文件
	logFile, err := os.Open("/Users/wwb/Desktop/code/go/go_debug/file/room_14_game_gift_log.txt")
	if err != nil {
		fmt.Printf("无法打开日志文件: %v\n", err)
		return
	}
	defer logFile.Close()

	// 获取文件信息
	fileInfo, err := logFile.Stat()
	if err != nil {
		fmt.Printf("无法获取文件信息: %v\n", err)
		return
	}

	// 读取整个文件内容
	data := make([]byte, fileInfo.Size())
	_, err = logFile.Read(data)
	if err != nil {
		fmt.Printf("无法读取文件内容: %v\n", err)
		return
	}

	// 分割JSON条目（假设每行一个JSON对象）
	lines := splitJSONLines(data)

	// 创建CSV文件
	csvFile, err := os.Create("parsed_logs.csv")
	if err != nil {
		fmt.Printf("无法创建CSV文件: %v\n", err)
		return
	}
	defer csvFile.Close()

	// 创建CSV写入器
	writer := csv.NewWriter(csvFile)
	defer writer.Flush()

	// 写入CSV标题行
	headers := []string{
		"gift_name", "total_amount", "total_discount", "ts_id", "receivers", "time",
	}
	if err := writer.Write(headers); err != nil {
		fmt.Printf("无法写入CSV标题行: %v\n", err)
		return
	}
	filterMap := map[int64]bool{}
	// 解析每条日志并写入CSV
	for _, line := range lines {
		if len(line) == 0 {
			continue
		}

		var entry Log14Entry
		if err := json.Unmarshal(line, &entry); err != nil {
			fmt.Printf("解析JSON失败: %v\n", err)
			continue
		}

		// 将receivers_map转换为JSON字符串
		receiversJSON, err := json.Marshal(entry.Data.Message.Receivers)
		if err != nil {
			fmt.Printf("转换receivers_map为JSON失败: %v\n", err)
			continue
		}
		if filterMap[entry.Data.Message.TsID] {
			continue
		}
		//把int类型time转成yyyymmddhhmm:ss
		timeStr := time.Unix(entry.Data.Message.Time, 0).Format("2006-01-02 15:04:05")
		// 写入CSV行
		record := []string{
			entry.Data.Message.GiftName,
			fmt.Sprintf("%d", entry.Data.Message.TotalAmount),
			fmt.Sprintf("%d", entry.Data.Message.TotalDiscount),
			fmt.Sprintf("%d", entry.Data.Message.TsID),
			string(receiversJSON),
			fmt.Sprintf("%s", timeStr),
		}
		filterMap[entry.Data.Message.TsID] = true
		if err := writer.Write(record); err != nil {
			fmt.Printf("写入CSV行失败: %v\n", err)
			continue
		}
	}

	fmt.Println("日志解析完成，CSV文件已生成")
}

// 分割JSON行（处理可能的多行JSON或单行多个JSON）
func splitJSONLines(data []byte) [][]byte {
	// 简单实现：假设每行一个完整的JSON对象
	// 更健壮的实现需要考虑JSON对象跨多行的情况
	var lines [][]byte
	var currentLine []byte
	var inString bool
	var escape bool

	for _, b := range data {
		// 处理JSON字符串中的引号和转义字符
		if b == '"' && !escape {
			inString = !inString
		}
		escape = (b == '\\' && !escape)

		// 遇到换行符且不在字符串内时分割行
		if b == '\n' && !inString {
			if len(currentLine) > 0 {
				lines = append(lines, currentLine)
				currentLine = nil
			}
		} else {
			currentLine = append(currentLine, b)
		}
	}

	// 添加最后一行
	if len(currentLine) > 0 {
		lines = append(lines, currentLine)
	}

	return lines
}
