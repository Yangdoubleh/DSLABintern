```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

#데이터 입력
input_data = np.arange(0, 2*np.pi, 0.1)
correct_data = np.cos(input_data)
input_data = (input_data) / np.pi
n_data = len(correct_data)

n_in = 1     #입력층 뉴런 수
n_mid = 3    #은닉층 뉴런 수
n_out = 1    #출력층 뉴런 수

wb_width = 0.01    #표준편차 설정
eta = 0.1          #학습률
epoch = 2001       #학습횟수
interval = 200     #학습 진행을 보는 간격

#출력층 클래스
class OutputLayer:
    def __init__(self, n_upper, n):
        #가중치와 편향값을 평균 0, 표준편차 0.01인 난수로 생성
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)
        
    def forward(self, x):
        #입력을 받아 가중치를곱하고 편향을 더한 후 활성화함수(항등함수)에 대입
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u
        
    def backward(self, t):
        delta = self.y - t  #출력층 delta = 출력층의 입력(self.y-t/오차) * 활성화함수 미분(1)
        
        #가중치, 편향, 출력의 기울기(공식)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        
    def update(self, eta):
        #확률적 경사하강법을 통한 학습
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
        
#은닉층 클래스
class MiddleLayer:
    def __init__(self, n_upper, n):
        #가중치와 편향값을 평균 0, 표준편차 0.01인 난수로 생성
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)
        
    def forward(self, x):
        #입력을 받아 가중치를곱하고 편향을 더한 후 활성화함수(시그모이드함수)에 대입
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1/(1+np.exp(-u))
        
    def backward(self, grad_y):
        delta = grad_y * (1-self.y) * (self.y) #은닉층 delta = 입력(grad_y) * 활성화함수 미분(1-self.y * self.y)
        
        #가중치, 편향, 출력의 기울기(공식)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis = 0)
        self.grad_x = np.dot(delta, self.w.T)
        
    def update(self, eta):
        #확률적 경사하강법을 통한 학습
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
        
#클래스를 이용해 은닉층, 출력층 구현
middle_layer = MiddleLayer(n_in, n_mid)
output_layer = OutputLayer(n_mid, n_out)

for i in range (epoch):
    
    index_random = np.arange(n_data)
    np.random.shuffle(index_random)

    total_error = 0
    plot_x = []
    plot_y = []
    
    #학습 과정(1에포크)
    for idx in index_random:
        
        x = input_data[idx:idx+1]
        t = correct_data[idx:idx+1]
        
        #순전파
        middle_layer.forward(x.reshape(1,1))
        output_layer.forward(middle_layer.y)
        
        #역전파
        output_layer.backward(t.reshape(1,1))
        middle_layer.backward(output_layer.grad_x)
        
        #가중치와 편향 수정
        middle_layer.update(eta)
        output_layer.update(eta)
        
        if i%interval == 0:
            
            y = output_layer.y.reshape(-1)
            
            total_error += 1.0/2.0 * np.sum(np.square(y - t))
            
            plot_x.append(x)
            plot_y.append(y)
            
    if i%interval == 0:
        plt.plot(input_data, correct_data, linestyle = "dashed")
        plt.scatter(plot_x, plot_y, marker="+")
        plt.show()
        
        print("Epoch:" + str(i) + "/" + str(epoch))
```

