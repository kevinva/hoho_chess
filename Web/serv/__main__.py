
import json, os
from web.py_lib import auto_chess

def __dir__(request_, response_, route_args_):
	folder = route_args_['dir']
	try:
		content = os.listdir('web/'+folder)
		content = '\n'.join(content)
		return response_.write_response_OK_(content_type_='text/plain', content_=content, charset_='UTF-8')
	except Exception as ex:
		return response_.write_response_not_found_()

def home(request_, response_, route_args_):
	content = '<script>location.href="web/index.html"</script>'
	return response_.write_response_OK_(content_type_='text/html', content_=content, charset_='UTF-8')

def ajax_(request_, response_, route_args_):
	params_ = request_.params_
	assert 'data' in params_, '服务请求参数中缺少 data'
	board_key = json.loads(params_['data'])
	print(f'hoho: ajax_! board_key={board_key}')
	board = auto_chess._board_from_key(board_key)
	move = auto_chess.auto_move(board)
	if move is None:
		move = []
	json_ = json.dumps(move)
	return response_.write_response_JSON_OK_(json_)

def start_server_(port_, max_threads_):
	from .lib.http_ import Http_
	http_ = Http_(ip_='0.0.0.0', port_=port_, web_path_='web', max_threads_=max_threads_)
	http_.add_route_('/ajax', ajax_, 'GET')
	http_.add_route_('/ajax', ajax_, 'POST')
	http_.add_route_('/__dir__/{dir}', __dir__, 'GET')
	http_.add_route_('/__dir__/{dir}', __dir__, 'POST')
	http_.add_route_('/', home, 'GET')
	http_.start_()

start_server_(8000, 100)
