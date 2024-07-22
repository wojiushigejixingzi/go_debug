package request_live_api

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
)

func Test_live_feed(t *testing.T) {
	url := "http://dev.live.huajiao.com/feed/liveFeed?name=tag_pr_party&num=30&offset=0&refresh&userid=45868734&platform=ios&version=9.9.9&rand=0.9344949892533336&time=1720860646&guid=1cf3e9e0427e8c94a6544ac5b291eb3a"
	method := "GET"

	client := &http.Client{}
	req, err := http.NewRequest(method, url, nil)

	if err != nil {
		fmt.Println(err)
		return
	}
	req.Header.Add("host", "")
	req.Header.Add("User-Agent", "Apifox/1.0.0 (https://apifox.com)")
	req.Header.Add("Cookie", "token=YWECu.a.ZrnMtc7Xzg--AZZTIdGC6sEc7571;undefined")
	req.Header.Add("Accept", "*/*")
	req.Header.Add("Cache-Control", "no-cache")
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
