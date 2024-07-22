package huajiao

import (
	"encoding/csv"
	"fmt"
	"os"
)

func Test_main() {

}

func gener_csv() {
	//加载本地csv文件
	file, err := os.Open("D:/go/src/huajiao/1.csv")
	if err != nil {
		fmt.Print(err)
	}
	defer file.Close()
	//读取csv文件
	reader := csv.NewReader(file)
	//读取csv文件中的数据
	data, err := reader.ReadAll()
	if err != nil {
		fmt.Print(err)
	}
	//打印读取的数据

}
