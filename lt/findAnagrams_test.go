package lt

import (
	"fmt"
	"testing"
)

func Test_find(t *testing.T) {
	s := "ababababab"
	p := "aab"
	find(s, p)
	//findAnagrams(s, p)
}

func findAnagrams(s string, p string) []int {
	lenS := len(s)
	lenP := len(p)
	if lenS < lenP {
		return nil
	}

	var ans []int
	for slowIndex := 0; slowIndex < lenS; slowIndex++ {
		allTrue := setAllTrue(p, lenP)
		sting_s := string(s[slowIndex])
		if ok := allTrue[sting_s]; !ok {
			continue
		}
		var temp []string
		temp = append(temp, string(sting_s))
		allTrue[sting_s] = false
		for fastIndex := slowIndex + 1; fastIndex < lenS; fastIndex++ {
			fStr := string(s[fastIndex])
			if len(temp) >= lenP {
				break
			}
			if _, ok := allTrue[fStr]; ok {
				temp = append(temp, fStr)
				allTrue[fStr] = false
			} else {
				break
			}

		}
		if len(temp) == lenP && allIsFalse(allTrue) {
			ans = append(ans, slowIndex)
		}
	}
	fmt.Println("echo:", ans)
	return ans
}

func allIsFalse(pArr map[string]bool) bool {
	sign := true
	for _, v := range pArr {
		if v {
			sign = false
			break
		}
	}
	return sign
}

func setAllTrue(p string, len int) map[string]bool {
	mp := map[string]bool{}

	for i := 0; i < len; i++ {
		mp[string(p[i])] = true
	}
	return mp
}

func find(s, p string) (ans []int) {
	sLen, pLen := len(s), len(p)
	if sLen < pLen {
		return
	}

	var sCount, pCount [26]int
	for i, ch := range p {
		sCount[s[i]-'a']++
		pCount[ch-'a']++
	}
	if sCount == pCount {
		ans = append(ans, 0)
	}

	for i, ch := range s[:sLen-pLen] {
		sCount[ch-'a']--
		sCount[s[i+pLen]-'a']++
		if sCount == pCount {
			ans = append(ans, i+1)
		}
	}
	fmt.Println(ans)
	return

}
