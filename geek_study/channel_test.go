package geek_study

import (
	"fmt"
	"testing"
)

func Test_channel(t *testing.T) {
	ch1 := make(chan int, 3)
	ch1 <- 1
	ch1 <- 2
	ch1 <- 3
	//循环读取channel
	for len(ch1) > 0 {
		a := <-ch1
		fmt.Println(a, len(ch1))
	}

}
