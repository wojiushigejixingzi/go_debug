package test

import (
	"fmt"
	"testing"
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
func Test_a(t *testing.T) {
	main()
}

func main() {
	var peo = Student{}
	think := "bitch"
	fmt.Println(peo.Speak(think))
}
