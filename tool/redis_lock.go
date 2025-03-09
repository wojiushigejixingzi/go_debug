package tool

import (
	"fmt"
	"github.com/go-redsync/redsync/v4"
	"github.com/go-redsync/redsync/v4/redis/goredis/v9"
	goredislib "github.com/redis/go-redis/v9"
	"time"
)

func LockTest(cacheKey string) {
	// Create a pool with go-redis (or redigo) which is the pool redisync will
	// use while communicating with Redis. This can also be any pool that
	// implements the `redis.Pool` interface.
	client := goredislib.NewClient(&goredislib.Options{
		Addr: "localhost:6379",
	})
	pool := goredis.NewPool(client) // or, pool := redigo.NewPool(...)
	// Create an instance of redisync to be used to obtain a mutual exclusion
	// lock.
	rs := redsync.New(pool)

	// Obtain a new mutex by using the same name for all instances wanting the
	// same lock.
	mutexname := "wangwenboTest"
	redsync.WithExpiry(10 * time.Second)
	mutex := rs.NewMutex(mutexname,
		redsync.WithExpiry(10*time.Second),
		redsync.WithTries(10),
		redsync.WithRetryDelay(2*time.Second),
		redsync.WithDriftFactor(0.01),
		redsync.WithTimeoutFactor(0.05),
	)

	// Obtain a lock for our given mutex. After this is successful, no one else
	// can obtain the same lock (the same mutex name) until we unlock it.
	if err := mutex.Lock(); err != nil {
		fmt.Println("lock failed")
		return
	}
	fmt.Println("lock acquired", cacheKey, "current time:", time.Now().UnixNano())
	//休眠500ms
	time.Sleep(2000 * time.Millisecond)
	// Do your work that requires the lock.

	// Release the lock so other processes or threads can obtain a lock.
	if ok, err := mutex.Unlock(); !ok || err != nil {
		fmt.Println("unlock failed")
	}
	fmt.Println("lock released========", cacheKey)
}
