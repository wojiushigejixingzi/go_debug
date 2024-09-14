package demo

import (
	"fmt"
	"testing"
	"time"
)

func Test_panic_recover(t *testing.T) {
	defer println("defer")
	defer func() {
		recover()
	}()
	panic("panic")
	time.Sleep(time.Second)
	fmt.Print("end")
}
