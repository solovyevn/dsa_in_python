from collections import deque as queue

def josephus(n, m):
    circle = queue()
    i = 0
    while i < n:
        circle.append(i)
        i+=1
    print("Circle: %s" % circle)
    eliminated = [circle.popleft(), ]
    while len(circle):
      j = 1
      while j < m:
          circle.append(circle.popleft())
          j+=1
      j = 1
      eliminated.append(circle.popleft())
    return eliminated
          
    
