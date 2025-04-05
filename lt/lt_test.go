package lt

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"sync"
	"testing"
	"time"
)

// 字母异位
// 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
// 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
func Test_main(t *testing.T) {
	lengthOfLongestSubstring("abcabcbb")
}

func groupAnagrams(strs []string) [][]string {
	//声明一个hash存储排序后的字符串
	mp := map[string][]string{}
	for _, str := range strs {
		s := []byte(str)
		sort.Slice(s, func(i, j int) bool {
			return s[i] < s[j]
		})
		sortStr := string(s)
		mp[sortStr] = append(mp[sortStr], str)
	}
	res := [][]string{}
	for _, str := range mp {
		res = append(res, str)
	}
	return res
}

func longestConsecutive(nums []int) int {
	mp := map[int]bool{}
	for _, num := range nums {
		mp[num] = true
	}
	longestStreak := 0
	for num := range mp {
		if !mp[num-1] {
			currentNum := num
			currentStreak := 1
			for mp[currentNum+1] {
				currentNum++
				currentStreak++
			}
			longestStreak = max(longestStreak, currentStreak)
		}
	}
	return longestStreak
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

func Test_moveZeros(t *testing.T) {
	nums := []int{0, 1, 0, 3, 12}
	moveZeroes(nums)
	fmt.Println(nums)
}

// 移动零
func moveZeroes(nums []int) {
	slow := 0
	fmt.Println(nums)
	for fast := 0; fast < len(nums); fast++ {
		if nums[fast] != 0 {
			temp := slow
			nums[slow], nums[fast] = nums[fast], nums[slow]
			slow++
			fmt.Println(nums, temp, slow)
		}
	}
}

// 盛最多水的容器
func maxArea(height []int) int {
	maxArea := 0
	left := 0
	right := len(height)
	for left < right {
		heightLeft := height[left]
		heightRight := height[right]
		tempArea := (right - left) * min(heightRight, heightLeft)
		maxArea = max(maxArea, tempArea)
	}
	return maxArea
}

func min(i, j int) int {
	if i > j {
		return j
	}
	return i
}

// 三数之和
func threeSum(nums []int) [][]int {
	var res [][]int
	length := len(nums)
	if length < 3 {
		return res
	}
	sort.Ints(nums)
	for i := 0; i < length; i++ {
		if nums[i] > 0 {
			return res
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		L := i + 1
		R := length - 1
		for L < R {
			temp := nums[i] + nums[L] + nums[R]
			if temp == 0 {
				res = append(res, []int{nums[i], nums[L], nums[R]})
				for L < R && nums[L] == nums[L+1] {
					L++
				}
				for L < R && nums[R] == nums[R-1] {
					R--
				}
				L++
				R--
			} else if temp > 0 {
				R--
			} else {
				L++
			}
		}
	}
	return res
}

// 无重复字符的最长子串
func lengthOfLongestSubstring(s string) int {
	//if len(s) == 0 {
	//	return 0
	//}
	//mp := map[byte]int{}
	//maxLength := 0
	//left := 0
	//for right := 0; right < len(s); right++ {
	//	if index, ok := mp[s[right]]; ok {
	//		left = max(left, index+1)
	//		fmt.Println(index, left)
	//	}
	//	mp[s[right]] = right
	//	maxLength = max(maxLength, right-left+1)
	//}
	//return maxLength
	/*
		滑动窗口
	*/

	window := map[byte]int{}
	left, right, res := 0, 0, 0
	for right < len(s) {
		c := s[right]
		right++
		window[c]++
		for window[c] > 1 {
			d := s[left]
			left++
			window[d]--
		}
		res = max(res, right-left)
	}
	return res
}

/**
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。



示例 1:

输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
 示例 2:

输入: s = "abab", p = "ab"
���: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。

*/
// 找到字符串中所有字母异位词 (TODO)
func findAnagrams(s, t string) []int {
	need := map[byte]int{}
	window := map[byte]int{}
	for i := 0; i < len(t); i++ {
		need[t[i]]++
	}
	left, right, valid := 0, 0, 0
	var res []int
	for right < len(s) {
		c := s[right]
		right++
		if _, ok := need[c]; ok {
			window[c]++
			if window[c] == need[c] {
				valid++
			}
		}
		//判断左侧窗口是否要收缩
		for right-left >= len(t) {
			//当窗口符合条件时，把起始索引加入结果
			if valid == len(need) {
				res = append(res, left)
			}
			d := s[left]
			left++
			if _, ok := need[d]; ok {
				if window[d] == need[d] {
					valid--
				}
				window[d]--
			}
		}

	}
	return res
}

func findAnagrams2(s, p string) (ans []int) {
	cnt := [26]int{} // 统计 p 的每种字母的出现次数
	for _, c := range p {
		cnt[c-'a']++
	}
	left := 0
	for right, c := range s {
		c -= 'a'
		cnt[c]--         // 右端点字母进入窗口
		for cnt[c] < 0 { // 字母 c 太多了
			cnt[s[left]-'a']++ // 左端点字母离开窗口
			left++
		}
		if right-left+1 == len(p) { // s' 和 p 的每种字母的出现次数都相同
			ans = append(ans, left) // s' 左端点下标加入答案
		}
	}
	return
}

func Test_findAna(t *testing.T) {
	findAnagrams1("cbaebabacd", "abc")
}
func findAnagrams1(s string, p string) []int {
	cnt := [26]int{}
	for i := range p {
		cnt[p[i]-'a']++
	}
	ans := []int{}
	l, r := 0, 0
	cnt2 := [26]int{}
	for r < len(s) {
		cnt2[s[r]-'a']++
		r++
		if r-l == len(p) {
			if cnt == cnt2 {
				ans = append(ans, l)
			}
			cnt2[s[l]-'a']--
			l++
		}
	}
	return ans
}

type ListNode struct {
	Val  int
	Next *ListNode
}

/*
 * 链表
 */

// 回文链表
func isPalindrome(head *ListNode) bool {
	if head == nil {
		return true
	}
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	//反转slow
	slow = reverseList(slow)
	for slow != nil {
		if slow.Val != head.Val {
			return false
		}
		slow = slow.Next
		head = head.Next
	}
	return true
}

// 反转链表
func reverseList(head *ListNode) *ListNode {
	curr := head
	var prev *ListNode
	for curr != nil {
		temp := curr.Next
		curr.Next = prev
		prev = curr
		curr = temp
	}
	return prev
}

// 环形链表
func hasCycle(head *ListNode) bool {
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if fast == slow {
			return true
		}
	}
	return false
}

// 环形链表2-使用map判定
func detectCycle(head *ListNode) *ListNode {
	mp := map[*ListNode]bool{}
	p := head
	for p != nil {
		if _, ok := mp[p]; ok {
			return p
		}
		mp[p] = true
		p = p.Next
	}
	return nil
}

// 环形链表2-快慢指针
func detectCycleFastAndSlow(head *ListNode) *ListNode {
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if fast == slow {
			p := head
			if p != slow {
				p = p.Next
				slow = slow.Next
			}
			return p
		}
	}
	return nil
}

// 相交链表
func getIntersectionNode(heaA, headB *ListNode) *ListNode {
	if heaA == nil || headB == nil {
		return nil
	}
	pa, pb := heaA, headB
	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}
		if pb == nil {
			pb = heaA
		} else {
			pb = pb.Next
		}
	}
	return pa
}

// 合并两个有序链表---递归
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list2 == nil {
		return list1
	}
	if list1 == nil {
		return list2
	}
	if list1.Val <= list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	}
}

// 合并领个有序链表---迭代
func mergeTwoLists1(list1, list2 *ListNode) *ListNode {
	prehead := &ListNode{}
	result := prehead
	for list1 != nil && list2 != nil {
		if list1.Val <= list2.Val {
			result.Next = list1
			list1 = list1.Next
		} else {
			result.Next = list2
			list2 = list2.Next
		}
		result = result.Next
	}
	if list1 != nil {
		result.Next = list1
	} else {
		result.Next = list2
	}
	return prehead.Next
}

// 两数相加
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	head := &ListNode{}
	cur := head
	curry := 0
	for l1 != nil || l2 != nil || curry != 0 {
		sum := 0
		sum += curry
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}
		cur.Next = &ListNode{sum % 10, nil}
		cur = cur.Next
		curry = sum / 10
	}
	return head.Next
}

// 删除链表倒数第n个节点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{0, head}
	first := head
	length := 0
	for first != nil {
		length++
		first = first.Next
	}
	length -= n
	first = dummy
	for length > 0 {
		length--
		first = first.Next
	}
	first.Next = first.Next.Next
	return dummy.Next
}

/**
 * 两两交换链表中的节点
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	rest := head.Next.Next
	newHead := head.Next
	newHead.Next = head
	head.Next = swapPairs(rest)
	return newHead
}

// Definition for a Node.
type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

// 随机链表的复制
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	mp := map[*Node]*Node{}
	cur := head
	for cur != nil {
		mp[cur] = &Node{cur.Val, nil, nil}
		cur = cur.Next
	}
	cur = head
	for cur != nil {
		mp[cur].Next = mp[cur.Next]
		mp[cur].Random = mp[cur.Random]
		cur = cur.Next
	}
	return mp[head]
}

//148. 排序链表
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

func Test_sort_list(t *testing.T) {
	head := &ListNode{4, &ListNode{2, &ListNode{1, &ListNode{3, nil}}}}
	res := sortList(head)
	for res != nil {
		a := res.Val
		t.Log(a)
		res = res.Next

	}

}
func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	////找到链表的中间节点并断开，形成左右链表，并递归下探
	//midNode := &ListNode{}
	//midNode = findMiddleNode(head)
	//s := midNode.Next
	//midNode.Next = nil

	//找链表的中间节点
	var f, s = head, head
	var ps *ListNode
	for f != nil && f.Next != nil {
		f = f.Next.Next
		ps = s
		s = s.Next
	}
	// 拆分为两个链表（利用 ps，与题目《206. 反转链表》有异曲同工之处）
	ps.Next = nil

	leftNode := sortList(head)
	rightNode := sortList(s)
	//return 合并两个有序链表
	return mergeTwoSortList(leftNode, rightNode)
}

// 寻找中间节点
func findMiddleNode(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow := head
	fast := head
	var ps *ListNode
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		ps = slow
		slow = slow.Next
	}
	ps.Next = nil
	return slow
}

// 合并两个有序链表
func mergeTwoSortList(list1, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	curr := dummy
	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			curr.Next = list1
			list1 = list1.Next
		} else {
			curr.Next = list2
			list2 = list2.Next
		}
		curr = curr.Next
	}
	if list1 != nil {
		curr.Next = list1
	} else {
		curr.Next = list2
	}
	return dummy.Next
}

func Test_climbStairs(t *testing.T) {
	c := climbStairs(3)
	fmt.Println(c)
}

func climbStairs(n int) int {
	mp := map[int]int{}
	mp[1] = 1
	mp[2] = 2
	for i := 3; i <= n; i++ {
		mp[i] = mp[i-1] + mp[i-2]
	}
	return mp[n]
}

func Test_generate(t *testing.T) {
	a := generate(5)
	fmt.Println(a)
}

func generate(numRows int) [][]int {
	num := make([][]int, numRows)
	for i := range num {
		num[i] = make([]int, i+1)
		num[i][0] = 1
		num[i][i] = 1
		for j := 1; j < i; j++ {
			num[i][j] = num[i-1][j] + num[i-1][j-1]
		}
	}
	return num
}

func Test_fib(t *testing.T) {
	f := fib(2)
	fmt.Println(f)

}

func fib(n int) int {
	if n < 2 {
		return n
	}
	dp := map[int]int{}
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func word_slice(s string, wordDict []string) bool {
	wordDictSet := map[string]bool{}
	for _, w := range wordDict {
		wordDictSet[w] = true
	}
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for i := 1; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			if dp[j] && wordDictSet[s[j:i]] {
				dp[j] = true
				break
			}
		}
	}
	return dp[len(s)]
}

func TestSums(t *testing.T) {
	nums := []int{1, 5, 11, 5}
	res := canPartition(nums)
	fmt.Println(res)
}

func canPartition(nums []int) bool {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2
	dp := make([]bool, target+1)
	dp[0] = true
	for _, num := range nums {
		for j := target; j >= num; j-- {
			dp[j] = dp[j] || dp[j-num]
			fmt.Println(dp, j, num, dp[j], dp[j-num])
		}
	}
	return dp[target]
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func Test_inorder(t *testing.T) {
	node := TreeNode{Val: 1}

	node.Left = &TreeNode{Val: 2}
	node.Right = &TreeNode{Val: 3}

	res := inorderTraversalV1(&node)
	fmt.Println(res)
}
func inorderTraversal(root *TreeNode) (res []int) {
	stack := []*TreeNode{}
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, root.Val)
		root = root.Right
	}
	return
}
func inorderTraversalV1(root *TreeNode) (res []int) {
	var inoder func(node *TreeNode)
	inoder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inoder(node.Left)
		res = append(res, node.Val)
		inoder(node.Right)
	}
	inoder(root)
	return res
}
func sortedArrayToBST(nums []int) *TreeNode {
	Left := 0
	right := len(nums) - 1
	var order func(nums []int, Left int, right int) *TreeNode
	order = func(nums []int, Left int, right int) *TreeNode {
		if Left > right {
			return nil
		}
		mid := (Left + right) / 2
		root := &TreeNode{Val: nums[mid]}
		root.Left = order(nums, Left, mid-1)
		root.Right = order(nums, mid+1, right)
		return root
	}
	return order(nums, Left, right)
}

func flatten(root *TreeNode) (ans []int) {
	var dfs func(root *TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			return
		}
		ans = append(ans, root.Val)
		dfs(root.Left)
		dfs(root.Right)
	}
	dfs(root)
	return ans
}

func flatten1(root *TreeNode) {
	for root != nil {
		if root.Left != nil {
			root = root.Right
		} else {
			pre := root.Left
			for pre.Right != nil {
				pre = pre.Right
			}
			pre.Right = root.Right
			root.Right = root.Left
			root.Left = nil
			root = root.Right
		}
	}
}

func Test_dominantIndex(t *testing.T) {
	nums := []int{3, 6, 1, 0}
	res := dominantIndex(nums)
	fmt.Println(res)

}
func dominantIndex(nums []int) int {
	maxNum := -1
	maxIndex := 0
	for i, num := range nums {
		if num > maxNum {
			maxNum = num
			maxIndex = i
		}
	}
	for i, num := range nums {
		if i != maxIndex && maxNum < 2*num {
			return -1
		}
	}
	return maxIndex

}

func twoSum(nums []int, target int) []int {
	mp := map[int]int{}
	for index, value := range nums {
		if key, ok := mp[value]; ok {
			return []int{key, index}

		}
		mp[target-value] = index
	}
	return nil
}

func Test_generateParenthesis11(t *testing.T) {
	result := generateParenthesis11(3)
	fmt.Println(result)
}

func generateParenthesis11(n int) (res []string) {
	var order func(left, right int, current string)
	order = func(left, right int, current string) {
		if left == 0 && right == 0 {
			res = append(res, current)
			return
		}
		if left > right {
			return
		}
		if left > 0 {
			order(left-1, right, current+"(")
		}
		if right > 0 {
			order(left, right-1, current+")")
		}
	}
	order(n, n, "")
	return res
}

func Test_generateParenthesis(t *testing.T) {
	result := generateParenthesis(3)
	fmt.Println(result)
}

func generateParenthesis(n int) (res []string) {
	var order func(left, right int, current string)
	order = func(left, right int, current string) {
		if left == 0 && right == 0 {
			res = append(res, current)
			return
		}
		if left > right {
			return
		}
		if left > 0 {
			order(left-1, right, current+"(")
		}
		if right > 0 {
			order(left, right-1, current+")")
		}
	}
	order(n, n, "")
	return res
}

func Test_findKthLargest(t *testing.T) {
	nums := []int{3, 2, 3, 1, 2, 4, 5, 5, 6}
	a := findKthLargest(nums, 4)
	fmt.Println(a)
}

func findKthLargest(nums []int, k int) int {
	sort.Ints(nums)
	return nums[len(nums)-2]
}

func Test_singleNumber(t *testing.T) {
	nums := []int{4, 1, 2, 1, 2}
	res := singleNumber(nums)
	fmt.Println(res)
}
func singleNumber(nums []int) int {
	res := 0
	for _, num := range nums {
		res ^= num
	}
	return res
}

func Test_aaa(t *testing.T) {
	a := 0
	fmt.Println(a ^ 0)
}

func majorityElement(nums []int) int {
	le := len(nums)
	temp := nums[0]
	count := 1
	for i := 1; i < le; i++ {
		if nums[i] == temp {
			count++
		} else {
			count--
			if count == 0 {
				count++
				temp = nums[i]
			}
		}
	}
	return temp
}

func Test_longestConsecutive(t *testing.T) {
	nums := []int{100, 4, 200, 1, 3, 2}
	a := longestConsecutive1(nums)
	fmt.Println(a)
}
func longestConsecutive1(nums []int) int {
	mp := map[int]bool{}
	for _, v := range nums {
		mp[v] = true
	}
	longestConsecutive := 1
	for num := range mp {
		if !mp[num-1] {
			current_num := num
			tempStack := 1
			for mp[current_num+1] {
				current_num++
				tempStack++
			}
			longestConsecutive = max(longestConsecutive, tempStack)
		}
	}
	return longestConsecutive
}

func Test_reverse(t *testing.T) {
	a := reverse(-123)
	fmt.Println(a)

}
func reverse(x int) int {
	res := 0
	for x != 0 {
		if res < math.MinInt32/10 || res > math.MaxInt32/10 {
			return 0
		}
		temp := x % 10
		x /= 10
		res = res*10 + temp
	}
	return res
}

func twoSum1(nums []int, target int) []int {
	mp := map[int]int{}
	for index, value := range nums {
		if key, ok := mp[target-value]; ok {
			return []int{key, index}
		}
		mp[index] = value
	}
	return nil
}

func Test_strSort(t *testing.T) {

}
func str_sort() []int {
	s := "cbaebabacd"
	p := "abc"
	valid := 0
	left, right := 0, 0
	res := []int{}
	window := map[byte]int{}
	need := map[byte]int{}
	for i := 0; i < len(p); i++ {
		need[p[i]]++
	}
	for right < len(s) {
		c := s[right]
		right++
		if _, ok := need[c]; ok {
			window[c]++
			if window[c] == need[c] {
				valid++
			}
		}
		if right-left >= len(p) {
			d := s[left]
			if valid == len(need) {
				res = append(res, left)
			}
			if _, ok := need[d]; ok {
				if window[d] == need[d] {
					valid--
				}
				window[d]--
			}
			left++
		}

	}
	return res
}

func Test_isSubsequence(t *testing.T) {
	s := "abc"
	a := "ahbgdc"
	fmt.Println(isSubsequence(s, a))
}

func isSubsequence(s string, t string) bool {
	need := map[byte]int{}
	statc := []byte{}
	for i := 0; i < len(s); i++ {
		need[s[i]]++
		statc = append(statc, s[i])
	}
	for right := 0; right < len(t); right++ {
		if _, ok := need[t[right]]; ok && statc[0] == t[right] {
			statc = statc[1:]
		}
	}
	if len(statc) != 0 {
		return false
	}
	return true

}

/**
1679. K 和数对的最大数目
中等
相关标签
相关企业
提示
给你一个整数数组 nums 和一个整数 k 。

每一步操作中，你需要从数组中选出和为 k 的两个整数，并将它们移出数组。

返回你可以对数组执行的最大操作数。



示例 1：

输入：nums = [1,2,3,4], k = 5
输出：2
解释：开始时 nums = [1,2,3,4]：
- 移出 1 和 4 ，之后 nums = [2,3]
- 移出 2 和 3 ，之后 nums = []
不再有和为 5 的数对，因此最多执行 2 次操作。
示例 2：

输入：nums = [3,1,3,4,3], k = 6
输出：1
解释：开始时 nums = [3,1,3,4,3]：
- 移出前两个 3 ，之后nums = [1,4,3]
不再有和为 6 的数对，因此最多执行 1 次操作。
*/

func Test_maxOperations(t *testing.T) {
	maxOperations([]int{1, 2, 3, 4}, 5)
}

func maxOperations(nums []int, k int) int {
	mp := map[int]int{}
	count := 0
	for _, num := range nums {
		if _, ok := mp[k-num]; ok && mp[k-num] > 0 {
			count++
			mp[k-num]--
		} else {
			mp[num]++
		}
	}
	return count
}

func Test_permute(t *testing.T) {
	permute([]int{1, 2, 3})
}

func permute(nums []int) [][]int {
	n := len(nums)
	res := [][]int{}
	path := make([]int, n)
	on_path := make([]bool, n)
	var dfs func(int)
	dfs = func(i int) {
		if n == i {
			res = append(res, append([]int(nil), path...))
			return
		}
		for j, on := range on_path {
			if !on {
				path[i] = nums[j]
				on_path[j] = true
				dfs(i + 1)
				on_path[j] = false
			}
		}
	}
	dfs(0)
	fmt.Println(res)
	return res
}

func frequencySort(s string) string {
	mp := map[byte]int{}
	for i := 0; i < len(s); i++ {
		mp[s[i]]++
	}
	return ""
}

func Test_mergeAlternately(t *testing.T) {
	fmt.Println(mergeAlternately("abc", "defg"))
}

func mergeAlternately(word1 string, word2 string) string {
	str := ""
	word1Len := len(word1)
	word1Index := 0
	word2Len := len(word2)
	word2Index := 0
	for word1Len > 0 && word2Len > 0 {
		tempStr := string(word1[word1Index]) + string(word2[word2Index])
		str += tempStr
		word1Index++
		word2Index++
		word1Len--
		word2Len--

	}
	if word1Len > 0 {
		str += word1[word1Index : word1Len-1]
	}
	if word2Len > 0 {
		str += word2[word2Index:]
	}
	return str
}
func gcdOfStrings(str1 string, str2 string) string {
	if str1+str2 != str2+str1 {
		return ""
	}
	i := dfs(len(str1), len(str2))
	return str1[0:i]
}

func dfs(a, b int) int {
	if b == 0 {
		return a
	}
	return dfs(b, a%b)
}

func Test_sortColors(t *testing.T) {
	fmt.Println(sortColors([]int{2, 0, 2, 1, 1, 0}))
}

func sortColors(nums []int) []int {
	num0, num1, num2 := 0, 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			nums[num2] = 2
			num2++
			nums[num1] = 1
			num1++
			nums[num0] = 0
			num0++
		} else if nums[i] == 1 {
			nums[num2] = 2
			num2++
			nums[num1] = 1
			num1++
		} else {
			nums[num2] = 2
			num2++
		}
	}
	return nums
}
func Test_sortColors1(t *testing.T) {
	res := sortColors1([]int{2, 0, 2, 1, 1, 0})
	fmt.Println(res)
}

func sortColors1(nums []int) []int {
	l := len(nums) - 1
	num0, num1, num2 := l, l, l
	for i := l; i > 0; i-- {
		if nums[i] == 2 {
			nums[num2] = 2
			num2--
			nums[num1] = 1
			num1--
			nums[num0] = 0
			num0--
		} else if nums[i] == 1 {
			nums[num2] = 2
			num2--
			nums[num1] = 1
			num1--
		} else {
			nums[num2] = 2
			num2--
		}
	}
	return nums
}

func Test_subarrauSum(t *testing.T) {
	fmt.Println(subarraySum([]int{1, 1, 1}, 2))

}
func subarraySum(nums []int, k int) int {
	ans := 0
	mp := map[int]int{0: 1}
	s := 0
	for _, num := range nums {
		s += num
		ans += mp[s-k]
		mp[s]++
	}
	return ans
}

func Test_maxSubArray(t *testing.T) {
	//fmt.Println(maxSubArray([]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}))
	fmt.Println(maxSubArray1([]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}))
}
func maxSubArray(nums []int) int {
	preMax := 0
	currentMax := nums[0]
	for _, num := range nums {
		preMax = max(num, preMax+num)
		currentMax = max(currentMax, preMax)
	}
	return currentMax
}

func maxSubArray1(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}
	fmt.Println("nums", nums, "max:", max)
	return max
}

func Test_appendSlice(t *testing.T) {
	appendSlice()
}
func appendSlice() {
	s := []int{5}
	s = append(s, 7)
	s = append(s, 9)
	x := append(s, 11)
	y := append(s, 12)
	fmt.Println(s, len(s), cap(s), x, len(x), cap(x), y, len(y), cap(y))
	s = append(s, 10)
	s = append(s, 11)
	s = append(s, 12)
	fmt.Println(s, len(s), cap(s), x, len(x), cap(x), y, len(y), cap(y))
}

func Test_a(t *testing.T) {
	ww()
}

var c = make(chan int)
var a int

func f() {
	a = 1
	<-c
}
func ww() {
	go f()
	c <- 0
	print(a)
}

func rotate(nums []int, k int) {
	if k == 0 {
		return
	}

}
func Test_productExceptSelf(t *testing.T) {
	//fmt.Println(productExceptSelf([]int{1, 2, 3, 4}))
	productExceptSelf([]int{1, 2, 3, 4})
}

func productExceptSelf(nums []int) []int {
	res := []int{0: 1}
	for i := 1; i < len(nums); i++ {
		res = append(res, res[i-1]*nums[i-1])
	}
	fmt.Println(res)
	right := 1
	for i := len(nums) - 1; i >= 0; i-- {
		res[i] *= right
		fmt.Println(i, right, nums[i])
		right = right * nums[i]
	}
	fmt.Println(res)
	return res
}

func productExceptSelf1(nums []int) []int {
	length := len(nums)
	answer := make([]int, length)
	// answer[i] 表示索引 i 左侧所有元素的乘积
	// 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
	answer[0] = 1
	for i := 1; i < length; i++ {
		answer[i] = nums[i-1] * answer[i-1]
	}

	// R 为右侧所有元素的乘积
	// 刚开始右边没有元素，所以 R = 1
	R := 1
	for i := length - 1; i >= 0; i-- {
		// 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
		answer[i] = answer[i] * R
		// R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
		R *= nums[i]
	}
	return answer
}
func Test_merge(t *testing.T) {
	merge([][]int{{1, 3}, {2, 6}, {8, 10}, {15, 18}})
}

func merge(intervals [][]int) [][]int {
	var res [][]int
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	for i := 0; i < len(intervals); i++ {
		for j := i + 1; j < len(intervals); j++ {
			if intervals[i][1] > intervals[j][0] && intervals[i][0] < intervals[j][1] {
				intervals[j][0] = min(intervals[i][0], intervals[j][0])
				intervals[j][1] = max(intervals[i][1], intervals[j][1])
				intervals[i] = []int{}
			} else {
				res = append(res, intervals[i])
			}
		}
	}
	return res
}
func Test_searchInsert(t *testing.T) {
	fmt.Println(searchInsert([]int{1, 3, 5, 6}, 5))
}

func searchInsert(nums []int, target int) int {
	lenght := len(nums)
	for i := 0; i < lenght; i++ {
		if nums[i] == target {
			return i
		}
		if nums[i] > target {
			return i - 1
		}
	}
	return lenght
}

func Test_serchRange(t *testing.T) {
	fmt.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 8))
}
func searchRange(nums []int, target int) []int {
	length := len(nums)
	if length == 0 || (length == 1 && nums[0] != target) {
		return []int{-1, -1}
	}
	l := 0
	r := length - 1
	for l < r {
		min := (l + r) / 2
		if nums[min] >= target {
			r = min
		} else {
			l = min + 1
		}
	}
	if nums[r] != target {
		return []int{-1, -1}
	}
	for i := r; i < length; i++ {
		if nums[i] != target {
			return []int{r, i - 1}
		}
	}
	return []int{r, length - 1}
}
func Test_searchIndex(t *testing.T) {
	fmt.Println(searchIndex([]int{1, 3, 5, 6}, 2))
}

func searchIndex(nums []int, target int) int {
	l := 0
	r := len(nums) - 1
	for l <= r {
		midIndex := (l + r) / 2
		if nums[midIndex] > target {
			r = midIndex - 1
		} else if nums[midIndex] < target {
			l = midIndex + 1
		} else {
			return midIndex
		}
	}
	return len(nums)
}

func Test_findMin(t *testing.T) {
	findMin([]int{3, 4, 5, 6, 7, 8, 1, 2})
}
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] > nums[len(nums)-1] {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return nums[left]
}

func Test_search(t *testing.T) {
	fmt.Println(search([]int{4, 5, 6, 7, 0, 1, 2}, 0))
}

func search(nums []int, target int) int {
	length := len(nums)
	left, right := 0, length-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < nums[left] {
			//右边有序
			if nums[mid] < target && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		} else {
			//左边有序
			if nums[mid] > target && target >= nums[left] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		}
	}
	return -1
}

func findKthLargestHeap(nums []int, k int) int {
	hp := &Heap{size: k}
	for _, num := range nums {
		hp.Add(num)
	}
	return hp.arr[0]
}

type Heap struct {
	arr  []int
	size int
}

func (hp *Heap) Add(num int) {
	if len(hp.arr) < hp.size {
		hp.arr = append(hp.arr, num)
		for i := len(hp.arr) - 1; i > 0; {
			p := (i - 1) / 2
			if p >= 0 && hp.arr[p] > hp.arr[i] {
				hp.Swap(p, i)
				i = p
			} else {
				break
			}
		}
	} else if num > hp.arr[0] {
		hp.arr[0] = num
		hp.Down(0)
	}
}
func (hp *Heap) Swap(a, b int) {
	hp.arr[a], hp.arr[b] = hp.arr[b], hp.arr[a]
}
func (hp *Heap) Down(i int) {
	k := i
	left, right := 2*i+1, 2*i+2
	n := len(hp.arr)
	if left < n && hp.arr[k] > hp.arr[left] {
		k = left
	}
	if right < n && hp.arr[k] > hp.arr[right] {
		k = right
	}
	if i != k {
		hp.Swap(i, k)
		hp.Down(k)
	}
}

func Test_partitionLabels(t *testing.T) {
	partitionLabels("ababcbacadefegdehijhklij")
}
func partitionLabels(s string) []int {
	mark := [26]int{}
	for i, c := range s {
		mark[c-'a'] = i
	}
	start, end := 0, 0
	res := []int{}
	for i, c := range s {
		end = max(end, mark[c-'a'])
		if i == end {
			res = append(res, end-start+1)
			start = i + 1
		}
	}
	return res
}

var wg sync.WaitGroup
var name string

func Test_goroutine(t *testing.T) {
	name = "wangwenbo"
	wg.Add(1)
	go func() {
		fmt.Println("go before", name)
		defer wg.Done()
		getName()
	}()
	wg.Wait()
}
func getName() {
	fmt.Println("go after", name)
}

func Test_coinChangeGreedy(t *testing.T) {
	coinChangeGreedy([]int{186, 419, 83, 408}, 6249)
}
func coinChangeGreedy(coins []int, amt int) int {
	sort.Slice(coins, func(i, j int) bool {
		return coins[i] < coins[j]
	})
	i := len(coins) - 1
	count := 0
	for amt > 0 {
		for i > 0 && coins[i] > amt {
			i--
		}
		amt -= coins[i]
		count++
	}
	if amt != 0 {
		fmt.Println("无法找零")
		return 0
	}
	return count
}

func Test_blackTrack(t *testing.T) {
	nums := []int{1, 2, 3}
	res := make([][]int, len(nums))
	state := make([]int, 0)
	selected := make([]bool, len(nums))
	blackTrack(&state, &nums, &selected, &res)
}

func blackTrack(stage *[]int, choices *[]int, selected *[]bool, res *[][]int) {
	if len(*stage) == len(*choices) {
		*res = append(*res, append([]int{}, *stage...))
	}
	duplicated := make(map[int]bool)
	for i := 0; i < len(*choices); i++ {
		choice := (*choices)[i]
		if _, ok := duplicated[choice]; !ok && !(*selected)[i] {
			duplicated[choice] = true
			(*selected)[i] = true
			*stage = append(*stage, choice)
			blackTrack(stage, choices, selected, res)
			(*selected)[i] = false
			*stage = (*stage)[:len(*stage)-1]
		}
	}
}

// 有重复元素的全排列
func Test_subsetSumINaive(t *testing.T) {
	nums := []int{3, 4, 5}
	target := 8
	res := make([][]int, 0)
	stage := make([]int, 0) //子集
	total := 0
	subsetSumINaive(total, target, &stage, &nums, &res)
	fmt.Println("res", res)
}

func subsetSumINaive(total, target int, stage, choices *[]int, res *[][]int) {
	//如果子集和等于目标值
	if total == target {
		*res = append(*res, append([]int{}, *stage...))
		return
	}
	//遍历所有选择
	for i := 0; i < len(*choices); i++ {
		//剪枝，如果超过目标值，则跳过选择
		if total+(*choices)[i] > target {
			continue
		}
		//尝试做出选择
		*stage = append(*stage, (*choices)[i])
		//递归
		subsetSumINaive(total+(*choices)[i], target, stage, choices, res)
		*stage = (*stage)[:len(*stage)-1]
	}
}

// 无重复元素的全排列
func Test_subsetSumINaiveNoDuplicate(t *testing.T) {
	nums := []int{3, 4, 5}
	//对nums进行排序
	sort.Ints(nums)
	//结果
	res := make([][]int, 0)
	target := 8
	//当前状态
	stage := make([]int, 0)
	//起始点
	start := 0
	subsetSumINaiveNoDuplicate(start, target, &stage, &nums, &res)
	fmt.Println("res", res)
}

func subsetSumINaiveNoDuplicate(start, target int, stage, choices *[]int, res *[][]int) {
	//子集的和目标值相等则记录值
	if target == 0 {
		*res = append(*res, append([]int{}, *stage...))
		return
	}
	//如果不想等，则从start开始遍历
	for i := start; i < len(*choices); i++ {
		//如果当前元素大于目标值 直接返回
		if target-(*choices)[i] < 0 {
			break
		}
		//记录选择
		*stage = append(*stage, (*choices)[i])
		subsetSumINaiveNoDuplicate(i, target-(*choices)[i], stage, choices, res)
		//回溯
		*stage = (*stage)[:len(*stage)-1]
	}
}

/*
*
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的
子集
（幂集）。
解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
示例 1：
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
示例 2：
输入：nums = [0]
输出：[[],[0]]
*/
func Test_subsets(t *testing.T) {
	nums := []int{1, 2, 3}
	res := make([][]int, 0)
	stage := make([]int, 0)
	if len(nums) == 0 {
		res = append(res, []int{})
		return
	}
	subsets(0, &stage, &nums, &res)
	fmt.Println("res", res)
}
func subsets(start int, stage, choices *[]int, res *[][]int) {
	*res = append(*res, append([]int{}, *stage...))
	for i := start; i < len(*choices); i++ {
		*stage = append(*stage, (*choices)[i])
		subsets(i+1, stage, choices, res)
		*stage = (*stage)[:len(*stage)-1]
	}
}

var phoneNums = map[string]string{
	"2": "abc",
	"3": "def",
	"4": "ghi",
	"5": "jkl",
	"6": "mno",
	"7": "pqrs",
	"8": "tuv",
	"9": "wxyz",
}

/*
*
示例 1：

输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
示例 2：

输入：digits = ""
输出：[]
示例 3：

输入：digits = "2"
输出：["a","b","c"]
*/
func Test_letterCombinations(t *testing.T) {
	digits := "23"
	res := letterCombinations(digits)
	fmt.Println("res", res)
}

func letterCombinations(digits string) (res []string) {
	if len(digits) == 0 {
		return
	}
	var dfs func(int, string)
	dfs = func(index int, str string) {
		fmt.Println("index", index, "str", str)
		if index == len(digits) {
			res = append(res, str)
			return
		}
		for _, c := range phoneNums[string(digits[index])] {
			dfs(index+1, str+string(c))
		}
	}
	dfs(0, "")
	return
}

func letterCombinations1(digits string) (res []string) {
	if len(digits) == 0 {
		return
	}
	var dfs func(i int, str string)
	dfs = func(i int, str string) {
		if i == len(digits) {
			res = append(res, str)
			return
		}
		for _, c := range phoneNums[string(digits[i])] {
			dfs(i+1, str+string(c))
		}
	}
	dfs(0, "")
	return res
}

func Test_permute1(t *testing.T) {
	nums := []int{1, 2, 3}
	res := permute1(nums)
	fmt.Println("res", res)
}

func permute1(nums []int) [][]int {
	res := [][]int{}
	state := []int{}
	selected := make([]bool, len(nums))
	var dfs func(state *[]int, nums *[]int, selected *[]bool, res *[][]int)
	dfs = func(state *[]int, nums *[]int, selected *[]bool, res *[][]int) {
		//符合要求加入结果
		if len(*state) == len(*nums) {
			*res = append(*res, append([]int{}, *state...))
		}
		for i := 0; i < len(*nums); i++ {
			choise := (*nums)[i]
			//如果没有选择过，则执行逻辑
			if !(*selected)[i] {
				//记录选择状态
				(*selected)[i] = true
				//加入选择结果
				*state = append(*state, choise)
				//进行下一搜索
				dfs(state, nums, selected, res)
				//回退
				(*selected)[i] = false
				*state = (*state)[:len(*state)-1]
			}
		}
	}
	dfs(&state, &nums, &selected, &res)
	return res
}

/*
*
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
*/
func Test_combinationSum(t *testing.T) {
	candidates := []int{8, 7, 4, 3}
	target := 11

	res := combinationSum(candidates, target)
	fmt.Println("res", res)
}

func combinationSum(candidates []int, target int) [][]int {
	res := [][]int{}
	start := 0
	sort.Ints(candidates)
	var dfs func(start int, target int, path []int)
	dfs = func(start int, target int, path []int) {
		if target == 0 {
			res = append(res, append([]int{}, path...))
			return
		}
		for i := start; i < len(candidates); i++ {
			if candidates[i] > target {
				break
			}
			dfs(i, target-candidates[i], append(path, candidates[i]))
		}

	}
	dfs(start, target, []int{})
	return res
}

func combine(n int, k int) [][]int {
	var (
		path []int
		res  [][]int
	)
	var dfs func(startIndex int)
	dfs = func(startIndex int) {
		if len(path) == k {
			res = append(res, append([]int{}, path...))
			return
		}
		for i := startIndex; i < n; i++ {
			path = append(path, i)
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}
	dfs(1)
	return res
}

func inorderTraversal4(root *TreeNode) []int {
	res := []int{}
	if root == nil {
		return res
	}
	var dfs func(root *TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			return
		}
		res = append(res, root.Val)
		dfs(root.Left)
		dfs(root.Right)
	}
	dfs(root)
	return res
}

func Test_buildTree(t *testing.T) {
	preorder := []int{3, 9, 20, 15, 7}
	inorder := []int{9, 3, 15, 20, 7}
	var dfs func(preorder []int, inorder []int) *TreeNode
	dfs = func(preorder []int, inorder []int) *TreeNode {
		n := len(preorder)
		if n == 0 {
			return nil
		}
		leftSize := slices.Index(inorder, preorder[0])
		left := dfs(preorder[1:1+leftSize], inorder[:leftSize])
		right := dfs(preorder[1+leftSize:], inorder[1+leftSize:])
		return &TreeNode{preorder[0], left, right}
	}
	a := dfs(preorder, inorder)
	fmt.Println(a)

}

func Test_uniquePaths(t *testing.T) {
	res := uniquePaths(3, 7)
	fmt.Println(res)

}
func uniquePaths(m int, n int) int {
	res := make([][]int, m)
	for i := 0; i < m; i++ {
		res[i] = make([]int, n)
		res[i][0] = 1
	}
	for j := 0; j < n; j++ {
		res[0][j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			res[i][j] = res[i-1][j] + res[i][j-1]
		}
	}
	return res[m-1][n-1]
}

func Test_minPathSum(t *testing.T) {
	grid := [][]int{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}}
	res := minPathSum(grid)
	fmt.Println(res)
}

func minPathSum(grid [][]int) int {
	res := make([][]int, len(grid))
	for i := 0; i < len(grid); i++ {
		res[i] = make([]int, len(grid[0]))
	}
	for j := 0; j < len(grid[0]); j++ {
		if j == 0 {
			res[0][j] = grid[0][j]
		} else {
			res[0][j] = res[0][j-1] + grid[0][j]
		}
	}
	for i := 0; i < len(grid); i++ {
		if i == 0 {
			res[i][0] = grid[i][0]
		} else {
			res[i][0] = res[i-1][0] + grid[i][0]
		}
	}
	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[0]); j++ {
			res[i][j] = min(res[i-1][j], res[i][j-1]) + grid[i][j]
		}
	}
	return res[len(grid)-1][len(grid[0])-1]
}

func Test_longestPalindrome(t *testing.T) {
	fmt.Println(longestPalindrome("bb"))
}

func longestPalindrome(s string) string {
	maxLength := 0
	left := 0
	right := 0
	lengh := len(s)
	dp := make([][]bool, lengh)
	for i := 0; i < lengh; i++ {
		dp[i] = make([]bool, lengh)
	}
	for i := lengh - 1; i >= 0; i-- {
		for j := i; j < lengh; j++ {
			if s[i] == s[j] {
				if j-i <= 1 {
					dp[i][j] = true
				} else if dp[i+1][j-1] {
					dp[i][j] = true
				}
			}
			if dp[i][j] && j-i+1 > maxLength {
				maxLength = j - i + 1
				left = i
				right = j
			}
		}
	}
	return s[left : right+1]
}

// 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
//
// 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
//
// 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
// 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
//
// 示例 1：
//
// 输入：text1 = "abcde", text2 = "ace"
// 输出：3
// 解释：最长公共子序列是 "ace" ，它的长度为 3 。
// 示例 2：
//
// 输入：text1 = "abc", text2 = "abc"
// 输出：3
// 解释：最长公共子序列是 "abc" ，它的长度为 3 。
// 示例 3：
//
// 输入：text1 = "abc", text2 = "def"
// 输出：0
// 解释：两个字符串没有公共子序列，返回 0 。
func Test_longestCommonSubsequence(t *testing.T) {
	res := longestCommonSubsequence("abcde", "ace")
	fmt.Println(res)
}
func longestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i, c1 := range text1 {
		for j, c2 := range text2 {
			if c1 == c2 {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
			}
		}
	}
	return dp[m][n]
}

//
//func Test_spiralOrder(t *testing.T) {
//	res := spiralOrder([][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}})
//	fmt.Println(res)
//}
//
//func spiralOrder(matrix [][]int) []int {
//	l := len(matrix)
//	r := len(matrix[0])
//	res := []int{}
//	for i := 0; i < len(matrix); i++ {
//		j
//		for j := 0; j < len(matrix[0]); j++ {
//			res = append(res, matrix[i][j])
//		}
//	}
//}

func rotate1(matrix [][]int) {
	n := len(matrix)
	temp := make([][]int, n)
	for i := range temp {
		temp[i] = make([]int, n)
	}
	for i, row := range matrix {
		for j, v := range row {
			temp[j][n-1-i] = v
		}
	}
	copy(matrix, temp)
}

func twoSum4(nums []int, target int) []int {
	mp := map[int]int{}
	for k, v := range nums {
		if j, ok := mp[target-v]; ok {
			return []int{k, j}
		}
		mp[v] = k
	}
	return []int{}
}

func Test_canConstruct(t *testing.T) {
	var ransomNote string
	ransomNote = "aa"
	var magazine string
	magazine = "abbb"
	res := canConstruct(ransomNote, magazine)
	fmt.Println(res)

}

func canConstruct(ransomNote string, magazine string) bool {
	magazineMp := map[rune]int{}
	for _, c := range magazine {
		magazineMp[c]++
	}

	for _, c := range ransomNote {
		if _, ok := magazineMp[c]; !ok {
			return false
		}
		if magazineMp[c] <= 0 {
			return false
		}
		magazineMp[c]--
	}
	return true
}

func isPalindrome2(s string) bool {
	//剔除非字符串
	newStr := ""
	for i := 0; i < len(s); i++ {
		iRune := s[i]
		if iRune >= 'A' && iRune <= 'Z' {
			newStr += string(iRune - 'A' + 'a')
		}
		if iRune >= 'a' && iRune <= 'z' {
			newStr += string(iRune)
		}
	}
	fmt.Println(newStr)
	left, right := 0, len(newStr)-1
	for left < right {
		if newStr[left] != newStr[right] {
			return false
		}
	}
	return true
}

func Test_asdfadsf(t *testing.T) {

}

func maxVowels(s string, k int) int {
	a := "aeiou"
	mp := map[byte]bool{}
	for i, _ := range a {
		mp[s[i]] = true
	}
	left, right := 0, 0
	window := 0
	res := 0
	for right < len(s) {
		if _, ok := mp[s[right]]; ok {
			window++
		}
		if right-left+1 > k {
			if _, ok := mp[s[left]]; ok {
				window--
			}
			left++
		}
		res = max(res, window)
		right++

	}
	return res

}

func Test_findMaxAverage(t *testing.T) {
	fmt.Println(findMaxAverage([]int{1, 12, -5, -6, 50, 3}, 1))
}
func findMaxAverage(nums []int, k int) float64 {
	var res = -math.MaxFloat64
	total := float64(0)
	for i := 0; i < len(nums); i++ {
		total += float64(nums[i])
		if i < k-1 {
			continue
		}
		res = math.Max(res, total/float64(k))
		total -= float64(nums[i-k+1])
	}
	return res
}

func float64Max(i, j float64) float64 {
	if i > j {
		return i
	}
	return j
}

func Test_minnumRecolors(t *testing.T) {
	fmt.Println(minimumRecolors("WBBWWBBWBW", 1))
}

func minimumRecolors(blocks string, k int) int {
	minStep := math.MaxInt64
	temp := 0
	for i := 0; i < len(blocks); i++ {
		if blocks[i] == 'W' {
			temp++
		}
		if i >= k {
			if blocks[i-k] == 'W' {
				temp--
			}
		}
		if i >= k-1 {
			minStep = min(minStep, temp)
		}
	}
	return minStep
}

// customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], minutes = 3
func Test_maxSatisfied(t *testing.T) {
	customers := []int{1, 0, 1, 2, 1, 1, 7, 5}
	grumpy := []int{0, 1, 0, 1, 0, 1, 0, 1}
	minutes := 3
	a := maxSatisfied(customers, grumpy, minutes)
	fmt.Println(a)
}
func maxSatisfied(customers []int, grumpy []int, minutes int) (res int) {
	//移动窗口，非窗口内按照grumpy是否生气累加，窗口内直接累加
	tempSum := 0
	left, right := 0, 0
	for right < len(customers) {
		//窗口外
		for i := 0; i < left; i++ {
			if grumpy[i] == 0 {
				tempSum += customers[i]
			}
		}
		//窗口内
		for right-left < minutes {
			tempSum += customers[right]
			right++
			continue
		}

		for i := right; i < len(customers); i++ {
			if grumpy[i] == 0 {
				tempSum += customers[i]
			}
		}
		left++
		res = max(res, tempSum)
		tempSum = 0
	}
	return res
}

//
//type LRUCache struct {
//	size       int
//	capacity   int
//	cache      map[int]*LinkNode
//	head, tail *LinkNode
//}
//
//type LinkNode struct {
//	key, value int
//	pre, next  *LinkNode
//}
//
//func initLinkNode(key, value int) *LinkNode {
//	return &LinkNode{
//		key:   key,
//		value: value,
//	}
//}
//
//func Constructor(capacity int) LRUCache {
//	l := LRUCache{
//		capacity: capacity,
//		cache:    map[int]*LinkNode{},
//		head:     initLinkNode(0, 0),
//		tail:     initLinkNode(0, 0),
//	}
//	l.head.next = l.tail
//	l.tail.pre = l.head
//	return l
//}
//
//func (this *LRUCache) Get(key int) int {
//	if _, ok := this.cache[key]; ok {
//		node := this.cache[key]
//		this.moveToHead(node)
//		return node.value
//	}
//	return -1
//}
//
//func (this *LRUCache) Put(key int, value int) {
//	if _, ok := this.cache[key]; !ok {
//		node := initLinkNode(key, value)
//		this.cache[key] = node
//		this.addToHead(node)
//		this.size++
//		if this.size > this.capacity {
//			removeNode := this.delTailNode()
//			delete(this.cache, removeNode.key)
//			this.size--
//		}
//	} else {
//		node := this.cache[key]
//		node.value = value
//		this.moveToHead(node)
//	}
//}
//
//func (this *LRUCache) addToHead(node *LinkNode) {
//	node.pre = this.head
//	node.next = this.head.next
//	this.head.next.pre = node
//	this.head.next = node
//}
//
//func (this *LRUCache) moveToHead(node *LinkNode) {
//	this.delNode(node)
//	this.addToHead(node)
//}
//
//// 删除节点
//func (this *LRUCache) delNode(node *LinkNode) {
//	node.next.pre = node.pre
//	node.pre.next = node.next
//}
//
//// 删除尾节点
//func (this *LRUCache) delTailNode() *LinkNode {
//	node := this.tail.pre
//	this.delNode(node)
//	return node
//}

func TestLoog(t *testing.T) {
	a := make(chan bool)
	b := make(chan bool)
	c := make(chan bool)
	go func() {
		<-a
		fmt.Println("a")
		b <- true
	}()
	go func() {
		<-b
		fmt.Println("b")
		c <- true
	}()
	go func() {
		<-c
		fmt.Println("c")
		a <- true
	}()
	a <- true
	time.Sleep(time.Second * 5)

}

func levelOrder(root *TreeNode) [][]int {
	res := [][]int{}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		n := len(queue)
		resTemp := []int{}
		for i := 0; i < n; i++ {
			node := queue[0]
			queue = queue[:1]
			resTemp = append(resTemp, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		res = append(res, resTemp)
	}
	return res
}

func Test_sortArrayByParty(t *testing.T) {
	num := []int{648, 831, 560, 986, 192, 424, 997, 829, 897, 843}
	//num := []int{2, 4, 1, 3}
	fmt.Println(sortArrayByParityII(num))
}

func sortArrayByParityII(nums []int) []int {
	if len(nums) < 2 {
		return nums
	}
	even, odd := 0, 1
	for even < len(nums) && odd < len(nums) {
		for even < len(nums) && nums[even]%2 == 0 {
			even += 2
		}
		for odd < len(nums) && nums[odd]%2 == 1 {
			odd += 2
		}
		if even < len(nums) && odd < len(nums) {
			nums[even], nums[odd] = nums[odd], nums[even]
			even += 2
			odd += 2
		}
	}
	return nums
}
func check(a int) bool {
	return a%2 == 0
}

func Test_isHappy(t *testing.T) {
	fmt.Println(isHappy(19))
}

func isHappy(n int) bool {
	visited := make(map[int]bool)
	for {
		// 终止条件：发现快乐数或进入循环
		if n == 1 {
			return true
		}
		if visited[n] {
			return false
		}

		// 标记当前数字已访问
		visited[n] = true

		// 计算下一步数值
		n = calculateNext(n)
	}
}

// calculateNext 计算数字各位平方和
func calculateNext(number int) int {
	squareSum := 0
	for number > 0 {
		digit := number % 10
		squareSum += digit * digit
		number /= 10
	}
	return squareSum
}

func kthSmallest(root *TreeNode, k int) (ans *int) {
	deep := 0
	var dfs func(root *TreeNode, deep int)
	dfs = func(root *TreeNode, deep int) {
		if root == nil {
			return
		}
		dfs(root.Left, deep)
		deep++
		if deep == k {
			*ans = root.Val
			return
		}
		dfs(root.Right, deep)
	}
	dfs(root, deep)
	return
}

func Test_NumIslands(t *testing.T) {
	grid := [][]byte{{'1', '1', '1', '1', '0'}, {'1', '1', '0', '1', '0'}, {'1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0'}}
	fmt.Println(numIslands(grid))
}

func numIslands(grid [][]byte) int {
	ans := 0
	if len(grid) == 0 {
		return ans
	}
	height := len(grid)
	weight := len(grid[0])
	for i := 0; i < height; i++ {
		for j := 0; j < weight; j++ {
			fmt.Println(i, j, height, weight)
			if grid[i][j] == '1' {
				searchLand(i, j, grid)
				ans++
			}
		}
	}
	return ans
}
func searchLand(i, j int, grid [][]byte) {
	if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) || grid[i][j] != '1' {
		return
	}
	grid[i][j] = 0
	searchLand(i+1, j, grid)
	searchLand(i-1, j, grid)
	searchLand(i, j+1, grid)
	searchLand(i, j-1, grid)
}

func exist(board [][]byte, word string) bool {
	length := len(board)
	weigh := len(board[0])
	for i := 0; i < length; i++ {
		for j := 0; j < weigh; j++ {
			if word[0] == board[i][j] && searchWorld(i, j, word, board, 0) {
				return true
			}
		}
	}
	return false
}

func searchWorld(i, j int, word string, board [][]byte, index int) bool {
	if i < 0 || j < 0 || i >= len(board) || j >= len(board[0]) || board[i][j] == 0 {
		return false
	}
	if board[i][j] != word[index] {
		return false
	}
	if len(word)-1 == index {
		return true
	}
	temp := board[i][j]
	board[i][j] = 0
	a := searchWorld(i+1, j, word, board, index+1)
	b := searchWorld(i-1, j, word, board, index+1)
	c := searchWorld(i, j+1, word, board, index+1)
	d := searchWorld(1, j-1, word, board, index+1)
	board[i][j] = temp
	return a || b || c || d
}

var phoneNumberMapStr = map[int][]byte{
	2: {'a', 'b', 'c'},
	3: {'d', 'e', 'f'},
	4: {'g', 'h', 'i'},
	5: {'j', 'k', 'l'},
	6: {'m', 'n', 'o'},
	7: {'p', 'q', 'r', 's'},
	8: {'t', 'u', 'v'},
	9: {'w', 'x', 'y', 'z'},
}

func Test_letterCombinations2(t *testing.T) {
	fmt.Println(letterCombinations2("23"))
}

func letterCombinations2(digits string) []string {
	length := len(digits)
	res := []string{}
	if length == 0 {
		return res
	}
	var dfs func(index int, path string)
	dfs = func(index int, path string) {
		if index == length {
			res = append(res, path)
			return
		}
		num := int(digits[index] - '0')
		fmt.Println(num)
		for _, v := range phoneNumberMapStr[num] {
			dfs(index+1, path+string(v))
		}

	}
	dfs(0, "")
	return res
}

type MinStack struct {
	minS    []int
	commonS []int
}

func ConstructorStack() MinStack {
	return MinStack{
		minS:    []int{math.MaxInt64},
		commonS: []int{},
	}
}

func (this *MinStack) Push(val int) {
	this.commonS = append(this.commonS, val)
	minStackValue := this.minS[len(this.minS)-1]
	if val < minStackValue {
		this.minS = append(this.minS, val)
	}
}

func (this *MinStack) Pop() {
	this.commonS = this.commonS[:len(this.commonS)-1]
	this.minS = this.minS[:len(this.minS)-1]
}

func (this *MinStack) Top() int {
	return this.commonS[len(this.commonS)-1]
}

func (this *MinStack) GetMin() int {
	return this.minS[len(this.minS)-1]
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(val);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.GetMin();
 */
