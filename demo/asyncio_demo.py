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

def do_requests(session):
    return session.get('http://rap2api.taobao.org/app/mock/data/2267369')

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []

        for _ in range(0, 10):
            tasks.append(do_requests(session))

        results = await asyncio.gather(*tasks)
        for r in results:
            print(f'status code: {r.status}')


async def main2():
    await asyncio.sleep(1)
    print('hello')

if __name__ == '__main__':
    # asyncio.run(main())

    # main2()
    asyncio.run(main2())