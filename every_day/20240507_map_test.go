package every_day

import (
	"fmt"
	"testing"
)

type person struct {
	name string
}

func Test_mp(t *testing.T) {
	var m map[person]int
	p := person{"mike"}
	fmt.Println(m[p])
}
