package main

import (
	"flag"
	"fmt"
	"gocv.io/x/gocv"
	pb "tensorflow_serving/apis"
)

func main() {
	request := &pb.PredictRequest{}
	webcam, _ := gocv.VideoCaptureDevice(0)
	window := gocv.NewWindow("hello")
	for {
		webcam.Read(&img)
		window.IMShow(img)
		window.WaitKey(1)
	}
}
