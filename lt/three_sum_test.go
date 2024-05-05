package lt

import (
	"sort"
	"testing"
)

func Test_three_sum(t *testing.T) {
	nums := []int{-1, 0, 1, 2, -1, -4}
	threeSum(nums)
}

func threeSum(nums []int) {
	n := len(nums)
	sort.Ints(nums)
	var ans [][]int
	for i, x := range nums[:n-2] {
		if i > 0 && x == nums[i-1] {
			continue
		}
		if x+nums[i+1]+nums[i+2] > 0 {
			continue
		}
		if x+nums[n-1]+nums[n-2] < 0 {
			continue
		}
		j, k := i+1, n-1
		for j < k {
			x := x + nums[j] + nums[k]
			if x > 0 {
				k--
			} else if x < 0 {
				j++
			} else {
				ans = append(ans, []int{x, nums[j], nums[k]})
				for j++; j < k && nums[j] == nums[j-1]; {
					j++
				}
				for k--; j < k && nums[k] == nums[k+1]; {
					k--
				}
			}
		}
	}
}
func threeSum1(nums []int) (ret [][]int) {
	sort.Ints(nums)

	n := len(nums)
	for first := 0; first < n-2; first++ {
		// 跳过重复的元素
		if first > 0 && nums[first] == nums[first-1] {
			continue
		}

		// 提前减枝
		if nums[first] > 0 {
			return ret
		}
		if nums[first]+nums[n-2]+nums[n-1] < 0 {
			continue
		}
		if nums[first]+nums[first+1]+nums[first+2] > 0 {
			continue
		}

		// 在 [first+1, n-1] 中找到 sum 为 target 的一对元素
		target := -nums[first]
		second, third := first+1, n-1
		for second < third {
			sum := nums[second] + nums[third]
			if sum == target {
				ret = append(ret, []int{nums[first], nums[second], nums[third]})

				// move second&third to next different index
				second++
				third--
				for second < n && nums[second] == nums[second-1] {
					second++
				}
				for third >= 0 && nums[third] == nums[third+1] {
					third--
				}
			} else if sum < target {
				second++
			} else {
				third--
			}
		}
	}
	return ret
}
