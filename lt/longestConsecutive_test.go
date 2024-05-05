package lt

import (
	"fmt"
	"testing"
)

func TestB(test *testing.T) {
	nums := []int{1, 2, 3, 4, 22, 34, 45, 56, 57, 58, 59, 60}
	a := longestConsecutive(nums)
	fmt.Println("a", a)
}

func longestConsecutive(nums []int) int {
	mp := map[int]bool{}
	for _, num := range nums {
		mp[num] = true
	}
	res := 0
	for n := range mp {
		if mp[n-1] {
			continue
		}
		cnt := 1
		for mp[n+1] {
			cnt++
			n++
		}
		res = max(res, cnt)
	}
	return res
}

func max(i, j int) int {
	if i < j {
		return j
	}
	return i
}
