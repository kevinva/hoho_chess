
import json, os
import time
from web.py_lib import auto_chess
import torch
import torch.multiprocessing as mp
import queue

from model.hoho_utils import *
from model.hoho_agent import *
from model.hoho_mcts import *
from model.hoho_cchessgame import *
from model.hoho_config import *


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
		global hoho_game, hoho_mcts, hoho_agent
		global match_count, agent_updating, agent_update_accepted, agent_update_path, last_update_finish_time, win_count

		if agent_update_accepted and (agent_update_path is not None):
			hoho_agent.load_model_from_path(model_path)
			print(f'[{now_datetime()}]{LOG_TAG_SERV} Agent updated! version={hoho_agent.version}')

			agent_update_accepted = False
			agent_update_path = None

		hoho_mcts = MCTS(start_player=PLAYER_RED)
		hoho_game = CChessGame()

		match_count += 1
		win_player = data[2]
		if win_player == 'Red':
			win_count += 1

		state = hoho_game.state
		pi, action, reward_u = hoho_mcts.take_simulation(hoho_agent, hoho_game)
		min_u, max_u = hoho_mcts.tree_u_score_bound()
		_, z, _ = hoho_game.step(action, reward_u, max_u, min_u)
		hoho_replay_buffer.add(state, pi.tolist(), z)

		move = convert_my_action_to_webgame_move(action)
		print(f'[{now_datetime()}]{LOG_TAG_SERV} get red move={move}')

		json_ = json.dumps(move)
	else:
		start_time = time.time()

		board_key = data_board
		# print(f'hoho: [ajax_] board_key={board_key}')
		board = auto_chess._board_from_key(board_key)
		black_move = auto_chess.auto_move(board)
		print(f'[{now_datetime()}]{LOG_TAG_SERV} get black move={black_move}')   # 注意这里黑方走法，是已经翻转了棋盘
		if black_move is None:
			black_move = []
			print(f'[{now_datetime()}]{LOG_TAG_SERV} [Error] black_move is None! ')
		else:
			black_state = hoho_game.state
			black_action = convert_webgame_opponent_move_to_action(black_move)
		
			# 模型只关心红方，这里强制造一个确定性黑方走子策略
			black_pi = np.zeros((ACTION_DIM,))
			black_pi[ACTIONS_2_INDEX[black_action]] = 1.0

			black_reward_u = hoho_mcts.update_root_with_action(black_action)  # 独自更新MCTS的根节点，因为webgame选的black_action跟自己模型选的不一定一样
			black_min_u, black_max_u = hoho_mcts.tree_u_score_bound()
			black_next_state, black_z, _ = hoho_game.step(black_action, black_reward_u, black_max_u, black_min_u)
			hoho_replay_buffer.add(flip_board(black_state), flip_action_probas(black_pi).tolist(), -black_z)  # 注意：这里要翻转为红方走子，将黑方的经验作为红方。hoho_todo!

			# 这里得到黑方的走子，就可以马上开始跑我方的模型
			red_state = hoho_game.state
			red_pi, red_action, red_reward_u = hoho_mcts.take_simulation(hoho_agent, hoho_game)
			red_min_u, red_max_u = hoho_mcts.tree_u_score_bound()
			print(f'[{now_datetime()}]{LOG_TAG_SERV} red_reward_u = {red_reward_u}')
			red_next_state, red_z, _ = hoho_game.step(red_action, red_reward_u, red_max_u, red_min_u)
			hoho_replay_buffer.add(red_state, red_pi.tolist(), red_z)

			print(f'[{now_datetime()}]{LOG_TAG_SERV} black_state={black_state}, with action={black_action}, to state={black_next_state}')
			print(f'[{now_datetime()}]{LOG_TAG_SERV} red_state={red_state}, with action={red_action}, to state={red_next_state}')

			red_move = convert_my_action_to_webgame_move(red_action)
			print(f'[{now_datetime()}]{LOG_TAG_SERV} get red move={red_move}')
		json_data = {'Black': list(black_move), 'Red': list(red_move)}
		json_ = json.dumps(json_data)

		if hoho_replay_buffer.size() >= 100:
			hoho_replay_buffer.save({'model_version': hoho_agent.version})
			hoho_replay_buffer.clear()

		print(f'[{now_datetime()}]{LOG_TAG_SERV} replay buffer size: {hoho_replay_buffer.size()}')
		print(f'[{now_datetime()}]{LOG_TAG_SERV} {round_count} rounds / {match_count} matches | elapse: {(time.time() - start_time):.3f}s')

		win_rate = (win_count / match_count) if match_count > 0 else 0
		print(f'[{now_datetime()}]{LOG_TAG_SERV} model version: {hoho_agent.version} | win count: {win_count} | win rate: {win_rate}')
		print('========================================================')

	if not message_queue.empty():
		msg_info = message_queue.get()
		print(f'[{now_datetime()}]{LOG_TAG_SERV} thread message: {msg_info}')
		if msg_info.get(KEY_MSG_ID) == AGENT_MSG_ID_TRAIN_FINISH:
			pass
		elif msg_info.get(KEY_MSG_ID) == AGENT_MSG_ID_SELF_BATTLE_FINISH:
			agent_update_accepted = msg_info.get(KEY_AGENT_ACCEPT)
			agent_update_path = msg_info.get(KEY_MODEL_PATH)
			agent_updating = False
			last_update_finish_time = time.time()

	# hoho_test，暂时不训练
	# if (not agent_updating) and ((time.time() - last_update_finish_time) > 3600):
	# if not agent_updating:
	# 	train_agent()
	# 	agent_updating = True
			

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


def train_agent():
	print(f'[{now_datetime()}]{LOG_TAG_SERV} Start training!')

	train_thread = threading.Thread(target=train, args=(hoho_agent, message_queue), name='hoho_train_thread')
	train_thread.start()
	# train_thread.join()

	# mp.set_start_method('spawn')
	# train_proc = mp.Process(target=train, args=(hoho_agent, rb))
	# train_proc.start()
	# # train_proc.join()


def find_top_version_model_path():
	root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	model_dir_path = os.path.join(root_dir_path, 'output', 'models')
	if not os.path.exists(model_dir_path):
		return None
		
	result_path = None
	top_version = 0
	for filename in os.listdir(model_dir_path):
		name = filename.split('.')[0]
		items = name.split('_')
		if len(items) == 5:
			if int(items[4]) > top_version:
				top_version = int(items[4])
				result_path = os.path.join(model_dir_path, filename)
          
	return result_path


if __name__ == '__main__':
	win_count = 0
	match_count = 0
	last_update_finish_time = 0
	message_queue = queue.Queue()
	agent_updating = False
	agent_update_accepted = False
	agent_update_path = None
	hoho_mcts = None
	hoho_game = None
	hoho_replay_buffer = ReplayBuffer()

	hoho_agent = Player()
	model_path = find_top_version_model_path()
	if model_path is not None:
		hoho_agent.load_model_from_path(model_path)

	# hoho_step 1
	print(f'[{now_datetime()}]{LOG_TAG_SERV}[pid={os.getpid()}] start server!')
	start_server_(8000, 100)

	# hoho_test
	# train_agent()



	