package demo

import (
	"fmt"
	"testing"
	"time"
)

func Test_channel(t *testing.T) {
	//for i := 0; i < 10; i++ {
	//	go func(i int) {
	//		fmt.Println(i)
	//	}(i)
	//}
	//time.Sleep(time.Second * 5)
	ch := make(chan int, 10)
	go pump(ch) // pump hangs
	time.Sleep(time.Second * 5)
	fmt.Println(<-ch)
}

func pump(ch chan int) {
	for i := 1; ; i++ {
		ch <- i
	}
}
