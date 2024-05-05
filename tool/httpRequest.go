package tool

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
)

func HttpPostJson(url string, b []byte) string {
	params := bytes.NewReader(b)
	resp, err := http.Post(url, "Content-Type:application/json", params)
	defer resp.Body.Close()
	if err != nil {
		fmt.Printf("http error %s %s", url, err)
		return ""
	}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("http error", err)
		return ""
	}
	return string(body)
}
