package tool

import (
	"github.com/disintegration/imaging"
	"image/color"
	"net/http"
)

func IsBlackScreen(url string) (bool, error, int) {
	i := 0
	resp, err := http.Get(url)
	if err != nil {
		return false, err, i
	}
	defer resp.Body.Close()

	img, err := imaging.Decode(resp.Body)
	if err != nil {
		return false, err, i
	}

	// 缩放图片以降低像素数量
	img = imaging.Resize(img, 100, 0, imaging.Lanczos)

	// 获取缩放后图片的尺寸
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 设置检查像素的步长，每隔 step 步检查一个像素
	step := 10

	// 检查图片中部分像素是否为黑色
	for y := 0; y < height; y += step {
		for x := 0; x < width; x += step {
			i++
			pixel := img.At(x, y)
			// 转换为RGBA颜色模式
			r, g, b, a := color.RGBAModel.Convert(pixel).RGBA()
			if r != 0 || g != 0 || b != 0 || a != 65535 {
				return false, nil, i
			}
		}
	}

	// 如果检查的像素都是黑色，则返回 true
	return true, nil, i
}
