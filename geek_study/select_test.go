package geek_study

import (
	"fmt"
	"testing"
	"time"
)

func Test_fob(t *testing.T) {
	c := make(chan int)
	quit := make(chan int)
	fobpmacco(c, quit)
}

func fobpmacco(c, quit chan int) {
	x, y := 0, 1
	for {
		select {
		case c <- x:
			x, y = y, x+y
		case <-quit:
			fmt.Print("quit")
			return
		default:
			fmt.Print("default11")
			return
		}
	}
}

func Test_fob1(t *testing.T) {
	ch := make(chan int)
	select {
	case i := <-ch:
		println(i)
	default:
		println("default")

	}
}

func Test_fob2(t *testing.T) {
	ch := make(chan int)
	go func() {
		for range time.Tick(1 * time.Second) {
			ch <- 0
		}
	}()

	for {
		select {
		case <-ch:
			println("case1")
		case <-ch:
			println("case2")
		}
	}
}
