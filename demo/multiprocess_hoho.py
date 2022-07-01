import multiprocessing as mp
import time

def func2(args):
    x = args[0]
    y = args[1]

    time.sleep(1)
    return x - y

def run_pool():
    cpu_worker_num = 3
    process_args = [(1, 1), (9, 9), (4, 4), (3, 3)]

    print(f'| inputs: {process_args}')
    start_time = time.time()
    with mp.Pool(cpu_worker_num) as p:
        outputs = p.map(func2, process_args)
    print(f'| outputs: {outputs} TimeUsed: {time.time() - start_time}')

def run_not_by_mp():
    start_time = time.time()
    process_args = [(1, 1), (9, 9), (4, 4), (3, 3)]
    for args in process_args:
        output = func2(args)
        print(f'output: {output}')
    print(f'TimeUsed: {time.time() - start_time}')


def func_pipe1(conn, p_id):
    print(p_id)

    time.sleep(0.1)
    conn.send(f'{p_id}_send1')
    print(p_id, 'send1')

    time.sleep(0.1)
    conn.send(f'{p_id}_send2')
    print(p_id, 'send2')

    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)

    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)


def func_pipe2(conn, p_id):
    print(p_id)

    time.sleep(0.1)
    conn.send(p_id)
    print(p_id, 'send')
    time.sleep(0.1)
    rec = conn.recv()
    print(p_id, 'recv', rec)


def run_pipe():
    from multiprocessing import Process, Pipe

    conn1, conn2 = Pipe()

    process = [Process(target=func_pipe1, args=(conn1, 'I1')),
               Process(target=func_pipe2, args=(conn2, 'I2')),
               Process(target=func_pipe2, args=(conn2, 'I3')), ]

    [p.start() for p in process]
    print('| Main', 'send')
    conn1.send(None)
    print('| Main', conn2.recv())
    [p.join() for p in process]


def function1(id):
    print(f'id {id}')

def run_process():
    processes = [mp.Process(target=function1, args=(1,)),
               mp.Process(target=function1, args=(2,))]
    [p.start() for p in processes]
    [p.join() for p in processes]

def func1(i):
    time.sleep(1)
    print(f'args {i}')

def run_queue():
    queue = mp.Queue(maxsize=4)
    queue.put(True)
    queue.put([0, None, object])
    print('qsize: ', queue.qsize())
    print(queue.get())
    print(queue.get())
    print('qsize: ', queue.qsize())

    processes = [mp.Process(target=func1, args=(queue,)),
                 mp.Process(target=func1, args=(queue,))]
    [p.start() for p in processes]
    [p.join() for p in processes]             

if __name__ == '__main__':
    # run_process()
    # run_pool()
    # run_not_by_mp()
    # run_pipe()  # 有错，run不了
    run_queue()


