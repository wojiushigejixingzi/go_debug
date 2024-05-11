package lt

import (
	"sort"
	"testing"
)

//字母异位
//输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
//输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
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

//移动零
func moveZeroes(nums []int) {
	slow := 0
	for fast := 0; fast < len(nums); fast++ {
		if nums[fast] != 0 {
			nums[slow], nums[fast] = nums[fast], nums[slow]
			slow++
		}
	}
}

//盛最多水的容器
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

//三数之和
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

//无重复字符的最长子串
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

//找到字符串中所有字母异位词
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

type ListNode struct {
	Val  int
	Next *ListNode
}

/*
 * 链表
 */

//回文链表
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

//反转链表
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

//环形链表
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

//环形链表2-使用map判定
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

//环形链表2-快慢指针
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

//相交链表
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

//合并两个有序链表---递归
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

//合并领个有序链表---迭代
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

//两数相加
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

//删除链表倒数第n个节点
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

//Definition for a Node.
type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

//随机链表的复制
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

//寻找中间节点
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

//合并两个有序链表
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
