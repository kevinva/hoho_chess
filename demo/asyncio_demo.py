# import requests

# def do_requests():
#     resp = requests.get('http://rap2api.taobao.org/app/mock/data/2267369')
#     print(f'status code: {resp.status_code}')


# def main():
#     for _ in range(10):
#         do_requests()

# if __name__ == '__main__':
#     main()


import aiohttp
import asyncio
import threading
from datetime import datetime
import time

# def do_requests(session):
#     return session.get('http://rap2api.taobao.org/app/mock/data/2267369')

# async def main():
#     async with aiohttp.ClientSession() as session:
#         tasks = []

#         for _ in range(0, 10):
#             tasks.append(do_requests(session))

#         results = await asyncio.gather(*tasks)
#         for r in results:
#             print(f'status code: {r.status}')


# async def main2():
#     await asyncio.sleep(1)
#     print('hello')


# async def cancel_me():
#     print('cancel_me(): sleep')
#     try:
#         # Wait for 1 hour
#         await asyncio.sleep(3600)
#     except asyncio.CancelledError:
#         print('cancel_me(): cancel sleep')
#         raise
#     finally:
#         print('cancel_me(): after sleep')

# async def main():
#     print('main(): running')
#     # Create a "cancel_me" Task
#     task = asyncio.create_task(cancel_me())

#     # Wait for 5 second
#     print('main(): sleep')
#     await asyncio.sleep(5)

#     print('main(): call cancel')
#     task.cancel()
#     try:
#         await task
#     except asyncio.CancelledError:
#         print('main(): cancel_me is cancelled now')


# async def cancel_me():
#     print('cancel_me(): sleep')
#     try:
#         await asyncio.sleep(3600)
#     except asyncio.CancelledError:
#         print('cancel_me(): cancel sleep')
#         raise
#     finally:
#         print('cancel_me(): after sleep')


# async def main():
#     print('main(): running')
#     # Create a 'cancel_me' Task
#     task = asyncio.create_task(cancel_me())
#     # loop = asyncio.get_event_loop()
#     # task = loop.create_task(cancel_me())

#     # Wait for 60 second
#     print('main(): sleep')
#     await asyncio.sleep(5)

#     print('main(): call cancel')
#     task.cancel()

#     try:
#         await task
#     except asyncio.CancelledError:
#         print('main(): cancel_me is cancelled now')

def hard_work():
    print('thread id: ', threading.current_thread())
    time.sleep(10)

async def do_async_job():
    # await asyncio.sleep(2)
    # print(datetime.now().isoformat(), 'thread id: ', threading.current_thread())

    await asyncio.to_thread(hard_work)
    await asyncio.sleep(1)
    print('job done!')


async def main():
    # await do_async_job()
    # await do_async_job()
    # await do_async_job()

    job1 = do_async_job()
    job2 = do_async_job()
    job3 = do_async_job()
    await asyncio.gather(job1, job2, job3)

    # try:
    #     await asyncio.wait_for(do_async_job(), timeout=1)
    # except asyncio.TimeoutError:
    #     print('timeout')


if __name__ == '__main__':
    asyncio.run(main())
    # loop = asyncio.get_event_loop()
    # task = loop.create_task(main())
    # loop.run_until_complete(task)