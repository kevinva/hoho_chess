from concurrent.futures import thread
import tarfile
import threading
import time
import random
import math

balance = 100
lock = threading.Lock()

def change(num, counter):
    global balance
    for i in range(counter):
        lock.acquire()
        balance += num
        balance -= num
        lock.release()
        if balance != 100:
            print(f'balance={balance}')
            break

class MultiThread(threading.Thread):

    def __init__(self, threadName, num):
        threading.Thread.__init__(self)
        self.name = threadName
        self.num = num

    def run(self):
        for i in range(self.num):
            print(f'{threading.current_thread().getName()}\tnum={i}')
            delay = math.ceil(random.random() * 2)
            time.sleep(delay)

def printNum(idx):
    for num in range(idx):
        print(f'{threading.current_thread().getName()}\tnum={num}')
        delay = math.ceil(random.random() * 2)
        time.sleep(delay)

def run(taskName):
    print('任务：', taskName)
    time.sleep(2)
    print(f'{taskName} 任务执行完毕')



if __name__ == '__main__':
    # th1 = threading.Thread(target=printNum, args=(2,), name='thread1')
    # th2 = threading.Thread(target=printNum, args=(3,), name='thread2')

    # th1.start()
    # th2.start()
    # th1.join()
    # th2.join()

    # thr1 = MultiThread('thread1', 3)
    # thr2 = MultiThread('thread2', 2)
    # thr1.start()
    # thr2.start()
    # thr1.join()
    # thr2.join()
    # print(f'{threading.current_thread().getName()} 线程结束')

    # start_time = time.time()
    # for i in range(3):
    #     thr = threading.Thread(target=run, args=(f'task-{i}',))
    #     thr.setDaemon(True)   # 设置为守护线程，只要主线程执行完，不管子线程有无执行完，子线程都会销毁
    #     thr.start()
    # print(f'{threading.current_thread().getName()}线程结束，当前线程数量={threading.active_count()}')
    # print(f'消耗时间：{time.time() - start_time}')

    # change()
    # print(f'修改后的 balance={balance}')

    thr1 = threading.Thread(target=change, args=(100, 500000), name='t1')
    thr2 = threading.Thread(target=change, args=(100, 500000), name='t2')
    thr1.start()
    thr2.start()
    thr1.join()
    thr2.join()
    print(f'{threading.current_thread().getName()} 线程结束')
