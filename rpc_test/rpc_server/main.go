package main

import (
	"context"
	"fmt"
	"go_debug/rpc_test/pb"
	"google.golang.org/grpc"
	"log"
	"net"
	"sort"
)

type server struct {
	pb.UnimplementedExampleServiceServer
}

func (s *server) SayHello(ctx context.Context, in *pb.HelloRequest) (*pb.HelloResponse, error) {
	return &pb.HelloResponse{Message: "Hello " + in.Name}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	num := []int{1, 3, 5, 2, 4}
	fmt.Printf("", 1, 2)
	println(1, 2)
	sort.Ints(num)

	pb.RegisterExampleServiceServer(s, &server{})
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
