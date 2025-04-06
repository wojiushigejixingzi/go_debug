package demo

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestTimeDuration(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	go handler(ctx, 1500*time.Millisecond)
	select {
	case <-ctx.Done():
		fmt.Println("TestTimeDuration timeout")
	}
}

func handler(ctx context.Context, duration time.Duration) {
	select {
	case <-ctx.Done():
		fmt.Println("handler ctx done")

	case <-time.After(duration):
		fmt.Println("handler timeout ")
	}
}
