package lt

import "sort"

func groupAnagrams(strs []string) [][]string {
	mp := map[string][]string{}

	for _, str := range strs {
		s := []byte(str)
		sort.Slice(s, func(i, j int) bool {
			return s[i] < s[j]
		})
		sortStr := string(s)
		mp[sortStr] = append(mp[sortStr], str)
	}

	res := make([][]string, 0, len(mp))
	for _, v := range mp {
		res = append(res, v)
	}
	return res
}
