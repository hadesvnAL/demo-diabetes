#1. load du lieu va chia train val va test
import numpy 
from numpy import loadtxt#load du lieu tu file csv vao file txt dataset
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model

dataset = loadtxt("D:\CNN\pima-indians-diabetes.csv", delimiter=',')

X = dataset[:,0:8]#input-8 gia tri dau tien
y = dataset[:,8]#output- gia tri cuoi 0-1
#chia du lieu train val, 
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
#lenh chia cua skleann, truyen vao X, y de bo ra train va val, test size nen chia train va val, it cho test
#chia lan 2 cua du lieu ben tren
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
#Dense: là lớp,
# 16: là số unit,
# input_dim số độ dài của ma trận số tham số đầu vào, ham relu
#phi khu tuyen tinh dam bao hieu don gian la ko cho cac unit co các duong weidt phai
# tuyen tinh đi, sigmoid la ham su dung xac dinh trang thai co benh va ko co benh 
#chon sigmoid phu hop tuong bai, sequential=model xay dung keras

# model = Sequential()
# model.add(Dense(16, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()# chay kien truc model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_data=(X_val ,y_val))



# model.save("D:\CNN\hadesjena.h5")

# do su dung 2 bien nhi phan nen dung binary,
# loss mat mat trong train 
#Epoch: la so vong train
#Batch_size: số dl nhap vao 1 lan train lay ra bao nhieu du lieu
#Validation bien ktra
# tao file da train

model = load_model("D:\CNN\hadesjena.h5")
#load model
loss, acc = model.evaluate(X_test, y_test)
#Đánh giá trên tập test
print("Loss= ", loss)
print("Acc= ", acc)
#in thu tren tap loss va acc

# #du doan 1 nguoi nao do
# hang = 28 
# X_new = X[hang]
# print(X_new)
# y_new= y[hang]
# print(X_new)
# #tao bien 1 ng nao do so thu tu tren bang du lieu
# X_new = numpy.expand_dims(X_new, axis=0)
# #chuyen ve dinh dang numpy tao ra 1 vecto ma tran ng du doan
# y_predict = model.predict(X_new)
# hades = "Tieu duong (1)"
# #khai bao bien
# if y_predict <=0.5:
# 	hades = "Ko tieu duong (0)"
# print("gia tru du doan = ", hades)
# print("gia tri thuc te", y_new)

#tạo giá trị từ thông số bên ngoài nhập vào. 
X_new = (10,139,80,0,0,27.1,1.441,57,)
print(X_new)
#tao bien 1 ng nao do so thu tu tren bang du lieu
X_new = numpy.expand_dims(X_new, axis=0)
#chuyen ve dinh dang numpy tao ra 1 vecto ma tran ng du doan
y_predict = model.predict(X_new)
hades = "Tieu duong (1)"
#khai bao bien
if y_predict <=0.5:
	hades = "Ko tieu duong (0)"
print("gia tru du doan = ", hades)
