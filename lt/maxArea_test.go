package lt

import "testing"

func Test_maxArea(t *testing.T) {
	height := []int{1, 8, 6, 2, 5, 4, 8, 3, 7}
	maxArea(height)
}

func maxArea(height []int) int {
	left, right := 0, len(height)-1
	ans := 0
	for left < right {
		area := (right - left) * min(height[right], height[left])
		ans = max(ans, area)
		if height[left] < height[right] {
			left++
		} else {
			right--
		}
	}
	return ans
}

func min(i, j int) int {
	if i < j {
		return i
	}
	return j
}
