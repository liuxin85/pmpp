# Makefile for compiling 4-mandel.cu with OpenGL and GLUT

# 编译器
NVCC = nvcc

# 源文件与输出文件
SRC = ${wildcard *.cu}
TARGET = ${patsubst %.cu, %, $(SRC)}

# 编译与链接参数
LDFLAGS = -arch=sm_80 -diag-suppress 2464 -Wno-deprecated-gpu-targets -lglut -lGL -lGLU

# 默认目标
all: $(TARGET)


# 每个 .cu → 可执行文件规则
%: %.cu
	$(NVCC) $< -o $@ $(LDFLAGS)

# 清理所有生成的可执行文件
clean:
	rm -f $(TARGET)

.PHONY: all clean