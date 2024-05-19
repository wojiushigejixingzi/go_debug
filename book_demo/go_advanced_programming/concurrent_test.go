package go_advanced_programming

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

/**
*并发编程：https://chai2010.cn/advanced-go-programming-book/ch1-basic/ch1-06-goroutine.html

 */
// Test_lock
func Test_lock(t *testing.T) {
	var mu sync.Mutex

	mu.Lock()
	go func() {
		defer mu.Unlock()
		fmt.Println("你好, 世界")
	}()
	mu.Lock()
}

//chan
func Test_chan(t *testing.T) {
	done := make(chan int)
	go func() {
		fmt.Println("你好, 世界")
		done <- 1
	}()
	<-done
}

//buffer chan
func Test_buffer_channel(t *testing.T) {
	//开启10个goroutine
	done := make(chan int, 10)
	for i := 0; i < cap(done); i++ {
		go func() {
			fmt.Println("你好, 世界")
			done <- 1
		}()
	}
	//等待10个后台线程执行完成
	for i := 0; i < cap(done); i++ {
		<-done
	}
	fmt.Println("执行完毕")
}

//sync.WaitGroup
func Test_sync_wait_group(t *testing.T) {
	var wg sync.WaitGroup
	//开启n个后台线程
	for i := 0; i < 10; i++ {
		go func() {
			wg.Add(1)
			fmt.Println("你好")
			wg.Done()
		}()
	}
	wg.Wait()
}

/**
*生产者消费者模型
 */

func Test_producer_consumer(t *testing.T) {
	ch := make(chan int, 64) //成果队列
	go producer(3, ch)
	go producer(5, ch)
	go consumer(ch)
	time.Sleep(1 * time.Second)

	//使用control + c退出
	//sig := make(chan os.Signal, 1)
	//signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	//fmt.Printf("quit (%v)\n\n", <-sig)
}

func producer(factor int, out chan<- int) {
	for i := 0; ; i++ {
		out <- i * factor
	}
}

func consumer(in <-chan int) {
	for v := range in {
		fmt.Println(v)
	}
}

/**
发布订阅模型 todo
*/

func Test_more_chan_search(t *testing.T) {
	for i := 0; i < 10; i++ {
		a()

	}
}
func a() {
	ch := make(chan string, 32)

	go func() {
		ch <- searchByBing("golang")
	}()
	go func() {
		ch <- searchByGoogle("golang")
	}()
	go func() {
		ch <- searchByBaidu("golang")
	}()

	fmt.Println(<-ch)
}

func searchByBaidu(s string) string {
	time.Sleep(1 * time.Second)
	return "百度搜索结果为：golang是世界上最好的语言"
}
func searchByBing(s string) string {
	time.Sleep(1 * time.Second)
	return "Bing搜索结果为：golang是世界上最好的语言"
}
func searchByGoogle(s string) string {
	time.Sleep(1 * time.Second)
	return "Google搜索结果为：golang是世界上最好的语言"
}

//素数筛 todo

func Test_foreach(t *testing.T) {
	total, sum := 0, 0
	var wg sync.WaitGroup
	for i := 1; i <= 10; i++ {
		wg.Add(1)
		sum += i
		go func(i int) {
			total += i
			wg.Done()
		}(i)
	}
	wg.Wait()
	fmt.Printf("total:%d sum %d", total, sum)
}
