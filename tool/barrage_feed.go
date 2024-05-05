package tool

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

type LogEntry struct {
	Time string `json:"time"`
	Data struct {
		Offset     int `json:"offset"`
		LivePlayVo struct {
			LiveID     string `json:"liveId"`
			FromAction string `json:"fromAction"`
			Tjdot      string `json:"tjdot"`
		} `json:"livePlayVo"`
	} `json:"data"`
}

func Export_barrage_feed() {
	// Open the log file
	file, err := os.OpenFile("/Users/wwb/Desktop/BarrageFeed.txt", os.O_RDONLY, os.ModePerm)
	if err != nil {
		fmt.Println("Failed to open the log file:", err)
		return
	}
	defer file.Close()

	// Create the CSV file
	outputFile, err := os.Create("output.csv")
	if err != nil {
		fmt.Println("Failed to create the output file:", err)
		return
	}
	defer outputFile.Close()

	// Create CSV writer
	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	// Write CSV header
	headers := []string{"Time", "Offset", "LiveID", "FromAction", "UID"}
	writer.Write(headers)

	// Create JSON decoder
	decoder := json.NewDecoder(file)

	// Decode JSON logs
	for {
		var entry LogEntry
		if err := decoder.Decode(&entry); err != nil {
			if err.Error() == "EOF" {
				break
			}
			fmt.Println("Failed to decode log entry:", err)
			return
		}

		// Parse time string
		t, err := time.Parse("2006-01-02 15:04:05.999999", entry.Time)
		if err != nil {
			fmt.Println("Failed to parse time:", err)
			return
		}
		formattedTime := t.Format("2006-01-02 15:04:05")

		// Parse UID
		tjdotParts := strings.Split(entry.Data.LivePlayVo.Tjdot, "_")
		if len(tjdotParts) < 2 {
			fmt.Println("Failed to parse UID:", entry.Data.LivePlayVo.Tjdot)
			return
		}
		uid := tjdotParts[1]

		// Write to CSV
		row := []string{
			formattedTime,
			fmt.Sprintf("%d", entry.Data.Offset),
			entry.Data.LivePlayVo.LiveID,
			entry.Data.LivePlayVo.FromAction,
			uid,
		}
		writer.Write(row)
	}

	fmt.Println("Parsing completed. Results saved to output.csv")
}
