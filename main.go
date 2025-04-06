package main

import "fmt"

func main() {
	//huajiao.Check_ip()
	//tool.Export_barrage_feed()
	//循环调用debu两次
	//for i := 0; i < 2; i++ {
	//	debu()
	//}

	//HotTimeUse()
	//url := "http://image.huajiao.com/beef96add3d5a2133c311481ef8f3e63.jpg"
	//types := mime.TypeByExtension(url)
	//fmt.Println(types)
	//tool.ExampleClient()
	// 1.创建路由
	//r := gin.Default()
	//// 2.绑定路由规则，执行的函数
	//// gin.Context，封装了request和response
	//r.GET("/", func(c *gin.Context) {
	//	c.String(http.StatusOK, "hello World!")
	//})
	//// 3.监听端口，默认在8080
	//// Run("里面不指定端口号默认为8080")
	//r.Run(":8011")
	//var mu sync.Mutex

	mp := make([]int, 0, 100000) // 预分配容量
	ch := make(chan int, 100000)

	// 启动生产者goroutine
	go func() {
		for i := 0; i < 100000; i++ {
			ch <- i
		}
		close(ch) // 关闭通道
	}()

	// 消费并收集结果（单线程）
	for v := range ch {
		mp = append(mp, v)
	}

	fmt.Println(len(mp)) // 输出100000
}
