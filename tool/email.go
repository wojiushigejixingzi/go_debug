package tool

import (
	"github.com/jordan-wright/email"
	"log"
	"net/smtp"
)

func sendToUser() bool {
	e := email.NewEmail()
	e.From = "dj <838206954@qq.com>"
	e.To = []string{"838206954@qq.com"}
	e.Subject = "welcome go world"
	e.Text = []byte("Text Body is, of course, supported!")
	err := e.Send("smtp.126.com:25", smtp.PlainAuth("", "838206954@qq.com", "yyy", "smtp.126.com"))
	if err != nil {
		log.Fatal(err)
	}
	return true
}
