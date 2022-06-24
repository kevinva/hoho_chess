
import json, os
import time
from web.py_lib import auto_chess
import torch
import torch.multiprocessing as mp

from model.hoho_utils import *
from model.hoho_agent import *
from model.hoho_mcts import *
from model.hoho_cchessgame import *


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
	data = json.loads(params_['data'])
	data_board = data[0]
	round_count = data[1]
	json_ = None

	if data_board == 'Action!': # 开始！
		global hoho_game, hoho_mcts, match_count

		hoho_game = CChessGame()
		hoho_mcts = MCTS(start_player=PLAYER_RED)
		match_count += 1

		state = hoho_game.state
		pi, action = hoho_mcts.take_simulation(hoho_agent, hoho_game)
		_, z, _ = hoho_game.step(action)
		hoho_replay_buffer.add(state, pi.tolist(), z)

		move = convert_my_action_to_webgame_move(action)
		print(f'{LOG_TAG_SERV} [ajax_] get red move={move}')

		json_ = json.dumps(move)
	else:
		start_time = time.time()

		board_key = data_board
		# print(f'hoho: [ajax_] board_key={board_key}')
		board = auto_chess._board_from_key(board_key)
		black_move = auto_chess.auto_move(board)
		print(f'{LOG_TAG_SERV} [ajax_] get black move={black_move}')   # 注意这里黑方走法，是已经翻转了棋盘
		if black_move is None:
			black_move = []
			print(f'{LOG_TAG_SERV} [Error] black_move is None! ')
		else:
			black_state = hoho_game.state
			black_action = convert_webgame_opponent_move_to_action(black_move)
			black_pi, _ = hoho_mcts.take_simulation(hoho_agent, hoho_game, update_root=False)  # 黑方用webgame自身的action，所以不要自动更新根节点
			hoho_mcts.update_root_with_action(black_action)  # 独自更新MCTS的根节点，因为webgame选的black_action跟自己模型选的不一定一样
			black_next_state, black_z, _ = hoho_game.step(black_action)
			black_real_state = flip_board(black_state)  # 这里是黑方走子，所以要翻转为红方
			black_pi = flip_action_probas(black_pi)  # 同样策略要翻转为红方
			hoho_replay_buffer.add(black_real_state, black_pi.tolist(), black_z)

			# 这里得到黑方的走子，就可以马上开始跑我方的模型
			red_state = hoho_game.state
			red_pi, red_action = hoho_mcts.take_simulation(hoho_agent, hoho_game)
			red_next_state, red_z, _ = hoho_game.step(red_action)
			hoho_replay_buffer.add(red_state, red_pi.tolist(), red_z)

			print(f'{LOG_TAG_SERV} black_state={black_state}, with action={black_action}, to state={black_next_state}')
			print(f'{LOG_TAG_SERV} red_state={red_state}, with action={red_action}, to state={red_next_state}')

			red_move = convert_my_action_to_webgame_move(red_action)
			print(f'{LOG_TAG_SERV} [ajax_] get red move={red_move}')
		json_data = {'Black': list(black_move), 'Red': list(red_move)}
		json_ = json.dumps(json_data)

		if hoho_replay_buffer.size() >= 100:
			hoho_replay_buffer.save()
			hoho_replay_buffer.clear()

		print(f'{LOG_TAG_SERV} replay buffer size: {hoho_replay_buffer.size()}')
		print(f'{LOG_TAG_SERV}[pid={os.getpid()}] {round_count} rounds / {match_count} matches | Elapsed={(time.time() - start_time):.3f}s/round')
		print('========================================================')

			
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


def start_train_process(agent, replay_buffer):
	train_proc = mp.Process(target=train, args=(agent, replay_buffer))
	train_proc.start()
	train_proc.join()


if __name__ == '__main__':
	torch.manual_seed(0)
	mp.set_start_method('spawn')

	match_count = 0
	hoho_mcts = None
	hoho_game = None
	hoho_agent = Player()
	hoho_replay_buffer = ReplayBuffer()

	# hoho_step 1
	print(f'{LOG_TAG_SERV} start server! pid={os.getpid()}')
	start_server_(8000, 100)


	