package every_day

import (
	"fmt"
	"sync"
	"testing"
)

const N = 10

var wg = &sync.WaitGroup{}

func Test_sync_wait_group(t *testing.T) {
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			println(i)
		}(i)
	}
	wg.Wait()
}

func Test_slice(t *testing.T) {
	s := []string{"asong", "Golang梦工厂"}
	change_clice(s)
	fmt.Println("inner slice: ", s)
}

func change_clice(s []string) {
	//append追加元素
	//*s = append(*s, "Go语言")
	s = append(s, "Go语言")
	fmt.Println("out slice: ", s)
}

func Test_slice_1(t *testing.T) {
	//s := []int{1, 2, 3, 4, 5, 6}
	//Assign1(s)
	//fmt.Println(s) // (1)
	//array := [5]int{1, 2, 3, 4, 5}
	//Reverse0(array)
	//fmt.Println(array) // (2)
	//s := []int{1, 2, 3}
	//Reverse2(s)
	//fmt.Println(s) // (3)
	//var a []int
	//for i := 1; i <= 3; i++ {
	//	a = append(a, i)
	//}
	//Reverse2(a)
	//fmt.Println(a) // (4)

	var b []int
	for i := 1; i <= 3; i++ {
		b = append(b, i)
	}
	Reverse2(b)
	fmt.Println(b) // (5)

	var c []int
	for i := 1; i <= 3; i++ {
		c = append(c, i)
	}
	Reverse3(c)
	fmt.Println(c) // (5)
}
func Reverse3(s []int) {
	s = append(s, 999, 1000, 1001)
	for i, j := 0, len(s)-1; i < j; i++ {
		j = len(s) - (i + 1)
		s[i], s[j] = s[j], s[i]
	}
}

func Reverse2(s []int) {
	s = append(s, 999)
	for i, j := 0, len(s)-1; i < j; i++ {
		j = len(s) - (i + 1)
		s[i], s[j] = s[j], s[i]
	}
}
func Reverse0(s [5]int) {
	for i, j := 0, len(s)-1; i < j; i++ {
		j = len(s) - (i + 1)
		s[i], s[j] = s[j], s[i]
	}
}

func Assign1(s []int) {
	s = []int{6, 6, 6}
}
