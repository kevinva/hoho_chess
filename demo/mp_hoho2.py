import os
import multiprocessing as mp
import time

def foo(i):
    print('This is ', mp.current_process().name)
    print('module name: ', __name__)
    print('Parent pid: ', os.getppid())
    print('Current pid: ', os.getpid())
    print('--------------------')

lis = []

def foo2(i):
    lis.append(i)
    print('This is Process [', i, '] and lis is [', lis, '] and lis.address is [', id(lis), ']')


def func(i, temp):
    temp[0] += 100
    print(f'Process {i} 修改数组第一个元素后----> {temp[0]}')

def func2(i, dic):
    dic['num'] = 100 + i
    print(dic.items())

def func3(i, q):
    ret = q.get()
    print(f'进程{i}从队列里获取一个{ret}, 然后又向队列放入了一个{i}')
    q.put(i)


def func4(i, lis, lc):
    # lc.acquire()
    lis[0] = lis[0] - 1
    time.sleep(1)
    print('say hi ', lis[0])
    # lc.release()

def func5(args):
    time.sleep(1)
    print('正在执行进程 ', args)


def main():
    # for i in range(5):
    #     p = mp.Process(target=foo2, args=(i,))
    #     p.start()
    #     p.join()

    # temp = mp.Array('i', [1, 2, 3, 4])
    # for i in range(10):
    #     p = mp.Process(target=func, args=(i, temp))
    #     p.start()
    #     p.join()

    # dic = mp.Manager().dict()
    # for i in range(10):
    #     p = mp.Process(target=func2, args=(i, dic))
    #     p.start()
    #     p.join()

    # lis = mp.Queue(20)
    # lis.put(0)
    # for i in range(10):
    #     p = mp.Process(target=func3, args=(i, lis))
    #     p.start()

    # array = mp.Array('i', 1)
    # array[0] = 10
    # lock = mp.RLock()
    # for i in range(10):
    #     p = mp.Process(target=func4, args=(i, array, lock))
    #     p.start()
    #     p.join()  # join方法就相当于锁的功能

    p = mp.Pool(5)
    for i in range(60):
        p.apply_async(func=func5, args=(i,))
    p.close()
    p.join()


if __name__ == '__main__':
    main()
    print('End!!!')