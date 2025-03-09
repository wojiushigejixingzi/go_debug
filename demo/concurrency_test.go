package demo

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

var mu sync.Mutex
var chain string

func Test_main(t *testing.T) {
	main()
}

func main() {
	chain = "main"
	A()
	fmt.Println(chain)
}
func A() {
	mu.Lock()
	defer mu.Unlock()
	chain = chain + " --> A"
	B()
}
func B() {
	chain = chain + " --> B"
	C()
}
func C() {
	mu.Lock()
	defer mu.Unlock()
	chain = chain + " --> C"
}

// =======================
var mu1 sync.RWMutex
var count int

func Test_main1(t *testing.T) {
	main1()
}
func main1() {
	go A1()
	time.Sleep(2 * time.Second)
	mu1.Lock()
	defer mu1.Unlock()
	count++
	fmt.Println(count)
}
func A1() {
	mu1.RLock()
	defer mu1.RUnlock()
	B1()
}
func B1() {
	time.Sleep(5 * time.Second)
	C1()
}
func C1() {
	mu1.RLock()
	defer mu1.RUnlock()
}
