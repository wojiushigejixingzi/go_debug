package demo

import (
	"fmt"
	"sort"
	"testing"
)

func Test_sortmap(t *testing.T) {
	var m = map[string]int{
		"one":   4,
		"two":   2,
		"three": 3,
	}
	var keys []string
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Println("key:", k, "values:", m[k])
	}
}
