"""
如果输入0:直接输出0
符号：
1.第一项无正符号
2.之后项单独判定正负
系数 num[i]
1.如果非最后一项且为1不输出
2.如果为0不输出
x
1.如果系数为0不输出
2.如果为末尾项不输出
次方  ^len-i
1.如果为末尾项不输出
2.如果系数为0不输出
3.如果次方为1不输出

"""
n=int(input())
if n==0:
    print(input())
else:
    num=list(map(int,input().split()))
    for i in range(len(num)):
        if i==0:
            if num[i]<0:
                print("-",end="")
        else:
            if num[i]<0:
                print("-",end="")
            elif num[i]>0:
                print("+",end="")
        if num[i]!=0:
            if i==(len(num)-1):
                print(abs(num[i]),end="")
            elif abs(num[i])!=1:
                print(abs(num[i]),end="")
        if num[i]!=0 and i!=(len(num)-1):
            print("x",end="")
        if num[i]!=0 and i!=(len(num)-1) and i!=(len(num)-2):
            print("^"+str(len(num)-i-1),end="")

