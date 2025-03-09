package every_day

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

func Test_print(t *testing.T) {
	print()
}
func print() {
	numberChan := make(chan struct{})
	letters := make(chan struct{}) // 用于字母 goroutine 控制
	done := make(chan struct{})    // 用于通知主线程完成
	go func() {
		num := 1
		for i := num; i <= 26; i++ {
			<-numberChan
			fmt.Printf("%d%d", num, num+1)
			num += 2
			letters <- struct{}{}
		}
	}()
	go func() {
		for i := 'A'; i <= 'Z'; i += 2 {
			<-letters
			fmt.Printf("%c%c", i, i+1)
			numberChan <- struct{}{}
		}
		done <- struct{}{}
	}()
	numberChan <- struct{}{}
	<-done
}

func sumAdd(c, quit chan int) {
	x := 0
	for {
		select {
		case c <- x:
			x++
			fmt.Println("c <- x", x)
		case <-quit:
			fmt.Println("quit")
			return
		}
	}
}
func Test_a(t *testing.T) {
	c := make(chan int, 1)
	quit := make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			fmt.Println(<-c)
		}
		quit <- 0
	}()
	sumAdd(c, quit)
}

func Test_timer(t *testing.T) {
	ticker := time.NewTicker(time.Millisecond * 500)
	go func() {
		for t := range ticker.C {
			fmt.Println("Tick at", t)
		}
	}()
}

// 交替打印数字和字母
func Test_chan(t *testing.T) {
	letter, number := make(chan bool), make(chan bool)
	wait := sync.WaitGroup{}

	go func() {
		i := 1
		for {
			select {
			case <-number:
				fmt.Print(i)
				i++
				letter <- true
			}
		}
	}()
	wait.Add(1)
	go func() {
		defer wait.Done()
		i := 'A'
		for {
			select {
			case <-letter:
				if i >= 'Z' {
					return
				}
				fmt.Print(string(i))
				i++
				number <- true
			}
		}
	}()
	number <- true
	wait.Wait()
}

func Test_chan2(t *testing.T) {
	letter, number := make(chan bool), make(chan bool)
	wg := &sync.WaitGroup{}
	go func() {
		i := 1
		for {
			select {
			case <-number:
				fmt.Print(i)
				i++
				letter <- true
			}
		}
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		i := 'A'
		for {
			select {
			case <-letter:
				fmt.Print(string(i))
				i++
				if i > 'Z' {
					return
				}
				number <- true
			}
		}
	}()
	number <- true
	wg.Wait()

}

func Test_zuseChannel(t *testing.T) {
	wg := sync.WaitGroup{}
	c := make(chan struct{})
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(num int, close <-chan struct{}) {
			defer wg.Done()
			<-close
			fmt.Println(num)
		}(i, c)
	}

	if WaitTimeout(&wg, time.Second*5) {
		close(c)
		fmt.Println("timeout exit")
	}
	time.Sleep(time.Second * 10)
}

func WaitTimeout(wg *sync.WaitGroup, timeout time.Duration) bool {
	ch := make(chan bool, 1)
	go time.AfterFunc(timeout, func() {
		ch <- true
	})
	go func() {
		wg.Wait()
		ch <- false
	}()
	return <-ch
}

func Test62(t *testing.T) {
	x := []string{"a", "b", "c"}
	fmt.Println(x[1:])
}

type MinStack struct {
	stack    []int
	minStack []int
}

func Constructor() MinStack {
	return MinStack{
		stack:    []int{},
		minStack: []int{math.MaxInt64},
	}
}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	minStackTop := this.minStack[len(this.minStack)-1]
	this.minStack = append(this.minStack, min(minStackTop, val))
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.minStack = this.minStack[:len(this.minStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

func Test_newmake(t *testing.T) {
	a := new(bool)
	b := make([]int, 0)
	//输出a值
	fmt.Println(*a)
	//输出b值
	fmt.Println(b)
}

func Test_defer(t *testing.T) {
	fmt.Println(c())
}

func c() (i int) {
	defer func() {
		i++
	}()
	return 1
}
