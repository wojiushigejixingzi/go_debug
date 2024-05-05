package lt

import "testing"

func Test(t testing.T) {
	TestTwoSum([]int{1, 2, 3, 4, 5, 6, 7, 8, 9}, 10)
}

func TestTwoSum(nums []int, target int) []int {
	numHash := map[int]int{}
	for i, num := range nums {
		if v, ok := numHash[target-num]; ok {
			return []int{v, i}
		}
		numHash[num] = i
	}
	return []int{}
}
