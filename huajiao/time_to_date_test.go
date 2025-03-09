package huajiao

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

type Data struct {
	Num                int64  `json:"num"`
	StartTime          int64  `json:"startTime"`
	EndTime            int64  `json:"endTime"`
	StartTimeFormatted string `json:"startTimeFormatted"`
	EndTimeFormatted   string `json:"endTimeFormatted"`
}

func Test_time_to_date(t *testing.T) {
	time_to()
}

func time_to() {
	// JSON 数据
	jsonData := `[{
	"num": 205876,
	"startTime": 1737216000000,
	"endTime": 1737219600000
}, {
	"num": 169133,
	"startTime": 1737219600000,
	"endTime": 1737223200000
}, {
	"num": 124169,
	"startTime": 1737223200000,
	"endTime": 1737226800000
}, {
	"num": 94394,
	"startTime": 1737226800000,
	"endTime": 1737230400000
}, {
	"num": 75879,
	"startTime": 1737230400000,
	"endTime": 1737234000000
}, {
	"num": 66333,
	"startTime": 1737234000000,
	"endTime": 1737237600000
}, {
	"num": 62594,
	"startTime": 1737237600000,
	"endTime": 1737241200000
}, {
	"num": 63873,
	"startTime": 1737241200000,
	"endTime": 1737244800000
}, {
	"num": 78264,
	"startTime": 1737244800000,
	"endTime": 1737248400000
}, {
	"num": 97727,
	"startTime": 1737248400000,
	"endTime": 1737252000000
}, {
	"num": 119816,
	"startTime": 1737252000000,
	"endTime": 1737255600000
}, {
	"num": 131427,
	"startTime": 1737255600000,
	"endTime": 1737259200000
}, {
	"num": 147823,
	"startTime": 1737259200000,
	"endTime": 1737262800000
}, {
	"num": 157032,
	"startTime": 1737262800000,
	"endTime": 1737266400000
}, {
	"num": 173258,
	"startTime": 1737266400000,
	"endTime": 1737270000000
}, {
	"num": 180630,
	"startTime": 1737270000000,
	"endTime": 1737273600000
}, {
	"num": 191530,
	"startTime": 1737273600000,
	"endTime": 1737277200000
}, {
	"num": 190940,
	"startTime": 1737277200000,
	"endTime": 1737280800000
}, {
	"num": 199527,
	"startTime": 1737280800000,
	"endTime": 1737284400000
}, {
	"num": 213918,
	"startTime": 1737284400000,
	"endTime": 1737288000000
}, {
	"num": 234217,
	"startTime": 1737288000000,
	"endTime": 1737291600000
}, {
	"num": 247785,
	"startTime": 1737291600000,
	"endTime": 1737295200000
}, {
	"num": 250022,
	"startTime": 1737295200000,
	"endTime": 1737298800000
}, {
	"num": 242117,
	"startTime": 1737298800000,
	"endTime": 1737302399000
}]`

	var dataList []Data
	err := json.Unmarshal([]byte(jsonData), &dataList)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		return
	}

	// 格式化时间
	for i, data := range dataList {
		// 转换时间戳为 time.Time 对象
		startTime := time.Unix(0, data.StartTime*int64(time.Millisecond))
		endTime := time.Unix(0, data.EndTime*int64(time.Millisecond))

		// 格式化时间
		dataList[i].StartTimeFormatted = startTime.Format("2006-01-02 15:04:05")
		dataList[i].EndTimeFormatted = endTime.Format("2006-01-02 15:04:05")
	}

	// 输出结果
	for _, data := range dataList {
		fmt.Printf("Num: %d, StartTime: %s, EndTime: %s\n", data.Num, data.StartTimeFormatted, data.EndTimeFormatted)
	}
}
