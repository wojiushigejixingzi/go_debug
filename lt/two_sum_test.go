package lt

import (
	"fmt"
	"sort"
	"testing"
)

func Test_tow_sum(t *testing.T) {
	nums := []int{-1, 0, 1, 2, -1, -4}
	twoSum(nums)
}

func twoSum(nums []int) (ans [][]int) {
	n := len(nums)
	sort.Ints(nums)
	for i, x := range nums[:n-2] {
		if i > 0 && x == nums[i-1] {
			continue
		}
		if x+nums[i+1]+nums[i+2] > 0 {
			continue
		}
		if x+nums[n-2]+nums[n-1] < 0 {
			continue
		}
		j, k := i+1, n-1
		for j < k {
			s := nums[j] + nums[i] + nums[k]
			if s > 0 {
				k--
			} else if s < 0 {
				j++
			} else {
				ans = append(ans, []int{x, nums[j], nums[k]})
				for j++; j < k && nums[j] == nums[j-1]; {
					j++
				}
				for k--; k > j && nums[k] == nums[k+1]; {
					k--
				}
			}
		}
	}
	fmt.Println(ans)
	return
}
