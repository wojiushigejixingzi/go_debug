package tool

import (
	"fmt"
	"google.golang.org/protobuf/proto"
	"testing"
)

func Test_user(t *testing.T) {
	user := &User{
		Username: "test",
		Age:      18,
	}
	//转换成protobuf格式
	marshal, err := proto.Marshal(user)
	if err != nil {
		panic(err)
	}
	newUser := &User{}
	err = proto.Unmarshal(marshal, newUser)
	if err != nil {
		panic(err)
	}
	fmt.Println(newUser.String())
}
