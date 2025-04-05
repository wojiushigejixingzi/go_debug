package main

import (
	"github.com/gin-gonic/gin"

	"net/http"
)

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
	r := gin.Default()
	// 2.绑定路由规则，执行的函数
	// gin.Context，封装了request和response
	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "hello World!")
	})
	// 3.监听端口，默认在8080
	// Run("里面不指定端口号默认为8080")
	r.Run(":8011")
}
