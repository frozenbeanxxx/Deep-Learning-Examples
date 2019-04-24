
conv1 = 7*7*3*32
conv2 = 3*3*32*64
dense1 = 3*3*64*128
dense2 = 128*64
softmax = 64*2
bias = 32 + 64 + 128 + 2
sum = conv1+conv2+dense1+dense2+softmax+bias
sum = sum * 4
print(conv1)
print(conv2)
print(dense1)
print(dense2)
print(sum)