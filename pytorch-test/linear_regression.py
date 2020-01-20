import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 

def linear_regression():
    # 线性的问题可以直接用一个全连接拟合
    # nn.Linear可以直接用来定义model
    input_size = 1
    output_size = 1
    num_epochs = 60
    lr = 0.001

    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    model = nn.Linear(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # aaa = torch.from_numpy(x_train)
    # bbb = aaa.numpy()
    # print(bbb)
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
 
    predicted = model(torch.from_numpy(x_train)).detach().numpy()
    plt.plot(x_train, y_train, 'ro', label='O')
    plt.plot(x_train, predicted, label='F')
    plt.legend()
    plt.show()

    model_weight = 'model/model_2.ckpt'
    torch.save(model.state_dict(), model_weight)

if __name__ == "__main__":
    linear_regression()