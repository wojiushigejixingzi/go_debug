package every_day

import (
	"fmt"
	"strings"
	"testing"
)

func Test_slice_apend(t *testing.T) {
	a := make([]int, 2000, 2001)
	b := make([]int, 1000, 1000)
	a = append(a, b...)
	fmt.Print(len(a), cap(a))
}

func Test_sliceappend(t *testing.T) {
	a := []int{1, 2, 3}
	b := []int{4, 5, 6}
	a = append(a, b...)
	fmt.Println(a)
}
func Test_build_tree(t *testing.T) {
	preorder := []int{3, 9, 20, 15, 7}
	inorder := []int{9, 3, 15, 20, 7}
	fmt.Println(buildTree(preorder, inorder))

}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	n := len(preorder)
	if n == 0 { // 空节点
		return nil
	}
	leftSize := Index(inorder, preorder[0]) // 左子树的大小
	left := buildTree(preorder[1:1+leftSize], inorder[:leftSize])
	right := buildTree(preorder[1+leftSize:], inorder[1+leftSize:])
	return &TreeNode{preorder[0], left, right}
}

func Index(slice []int, value int) int {
	for i, v := range slice {
		if v == value {
			return i
		}
	}
	return -1
}

func Test_subarrySym(t *testing.T) {
	nums := []int{1, 1, 1}
	k := 2
	fmt.Println(subarraySum(nums, k))
}
func subarraySum(nums []int, k int) (ans int) {
	s := 0
	cnt := map[int]int{0: 1} // s[0]=0 单独统计
	for _, x := range nums {
		s += x
		ans += cnt[s-k]
		cnt[s]++
	}
	return
}
func isValid(s string) bool {
	if len(s)%2 != 0 {
		return false
	}
	var stack []rune
	for _, i := range s {
		if i == '(' {
			stack = append(stack, ')')
		} else if i == '{' {
			stack = append(stack, '}')
		} else if i == '[' {
			stack = append(stack, ']')
		} else {
			if len(stack) == 0 || stack[len(stack)-1] != i {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}

func Test_slice_pop(t *testing.T) {
	stack := []int{1, 2, 3, 4, 5}
	s := stack[0:1]
	fmt.Println(s)
	stack = stack[len(stack)-1:]
	fmt.Println(stack)
}
func Test_decodeString(t *testing.T) {
	s := "3[a2[c]]"
	fmt.Println(decodeString(s))
}
func decodeString(s string) string {
	numStack := []int{}
	strStack := []string{}

	result := ""
	num := 0
	for _, char := range s {
		if char > '0' && char < '9' {
			num = num*10 + int(char-'0')
		} else if char == '[' {
			numStack = append(numStack, num)
			strStack = append(strStack, result)
			num = 0
			result = ""
		} else if char == ']' {
			count := numStack[len(numStack)-1]
			numStack = numStack[:len(numStack)-1]
			temp := strStack[len(strStack)-1]
			strStack = strStack[:len(strStack)-1]
			result = strings.Repeat(temp, count) + result
		} else {
			result += string(char)
		}
	}
	return result
}
