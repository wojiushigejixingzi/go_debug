package lt

import (
	"fmt"
	"testing"
)

func TestMoveZeroes(t *testing.T) {
	nums := []int{1, 10, 0, 9, 4, 0, 4, 0, 44, 0}
	moveZeroes(nums)
}

func moveZeroes(nums []int) {
	slowIndex := 0
	for fastIndex := 0; fastIndex < len(nums); fastIndex++ {
		if nums[fastIndex] != 0 {
			nums[slowIndex], nums[fastIndex] = nums[fastIndex], nums[slowIndex]
			slowIndex++
		}
	}
	fmt.Println(nums)
}
