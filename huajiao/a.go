package huajiao

import (
	"fmt"
)

type People interface {
	Speak(string) string
}

type Student struct{}

func (stu *Student) Speak(think string) (talk string) {
	if think == "sb" {
		talk = "你是个大帅比"
	} else {
		talk = "您好"
	}
	return
}

func aa() {
	var peo People = &Student{}
	think := "sb"
	fmt.Println(peo.Speak(think))
}

func Do() {
	aa()
}
