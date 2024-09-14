package huajiao

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
)

func Test_func(t *testing.T) {
	//并发处理
	for i := 0; i < 5; i++ {
		real_curl()
	}
}

func real_curl() {
	url := "http://live.test.huajiao.com/PublicRoomV2/Room/RoomInfo?room_id=422&userid=45865470&platform=ios&version=9.9.9&rand=0.385758081224604&time=1724914330&guid=4ef79fdb8fc85903cf41493a055b02fe"
	method := "GET"

	client := &http.Client{}
	req, err := http.NewRequest(method, url, nil)

	if err != nil {
		fmt.Println(err)
		return
	}
	req.Header.Add("User-Agent", "Apifox/1.0.0 (https://apifox.com)")
	req.Header.Add("Cookie", "token=YWECu9n.ZvemfAcT6g--2dUU7sT03pG3ffe3;undefined")
	req.Header.Add("Accept", "*/*")
	req.Header.Add("Cache-Control", "no-cache")
	req.Header.Add("Host", "live.test.huajiao.com")
	req.Header.Add("Connection", "keep-alive")

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
