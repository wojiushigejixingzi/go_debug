package tool

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func Test_redisLock(t *testing.T) {

	LockTest(fmt.Sprintf("wangwenbo:spinlock%d", 1))
	return

	//每50ms并发请求一次，缓存用"wangwenbo:spinlock"拼接累加次数（1，2，3，4，5.。。。。），并发请求10次
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			LockTest(fmt.Sprintf("wangwenbo:spinlock%d", i))
		}()
	}
	wg.Wait()
	fmt.Println("done")
}

func Lock(cacheName string) {
	time.Sleep(500 * time.Millisecond)
	fmt.Println("lock acquired", cacheName)
}
