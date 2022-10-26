
import json, os
import time
from web.py_lib import auto_chess
import torch
import torch.multiprocessing as mp
import queue
import numpy as np

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
		global hoho_game, hoho_mcts, hoho_agent, hoho_round
		global match_count, agent_updating, agent_update_accepted, agent_update_path, last_update_finish_time, win_count

		if agent_update_accepted and (agent_update_path is not None):
			hoho_agent.load_model_from_path(agent_update_path)
			LOGGER.info(f'Agent updated! version={hoho_agent.version}')
			
			agent_update_accepted = False
			agent_update_path = None

		hoho_mcts = MCTS(start_player=PLAYER_RED)
		hoho_game = CChessGame()

		match_count += 1
		win_player = data[2]

		if hoho_round is not None:
			if win_player == 'Red':
				win_count += 1
				hoho_round.update_winner('Red')
			elif win_player == 'Black':
				hoho_round.update_winner('Black')
			else:
				hoho_round.update_winner()

			if hoho_round.size() > 0:
				hoho_replay_buffer.add_round(hoho_round)

		hoho_round = Round(int(time.time()))

		if hoho_replay_buffer.size() >= 100:
			hoho_replay_buffer.save({'model_version': hoho_agent.version})
			hoho_replay_buffer.clear()

		state = hoho_game.state
		pi, action = hoho_mcts.take_simulation(hoho_agent, hoho_game)
		_, z, _ = hoho_game.step(action)
		hoho_round.add_red_step(state, pi.tolist())

		move = convert_my_action_to_webgame_move(action)
		LOGGER.info(f' get red move={move}')

		json_ = json.dumps(move)
	else:
		start_time = time.time()

		board_key = data_board
		# print(f'hoho: [ajax_] board_key={board_key}')
		board = auto_chess._board_from_key(board_key)
		black_move = auto_chess.auto_move(board)
		LOGGER.info(f'get black move={black_move}')  # 注意这里黑方走法，已经翻转了棋盘
		if black_move is None:
			black_move = []
			LOGGER.error('black_move is None!')
		else:
			black_state = hoho_game.state
			black_action = convert_webgame_opponent_move_to_action(black_move)

			# hoho_todo: 再验证black_move, black_action的正确性
		
			# 模型只关心红方，这里强制造一个确定性黑方走子策略
			black_pi = np.zeros((ACTION_DIM,))
			black_pi[ACTIONS_2_INDEX[black_action]] = 1.0

			hoho_mcts.update_root_with_action(black_action)  # 独自更新MCTS的根节点，因为webgame选的black_action跟自己模型选的不一定一样
			black_next_state, black_z, _ = hoho_game.step(black_action)
			hoho_round.add_black_step(flip_board(black_state), flip_action_probas(black_pi).tolist())  # 注意：这里要翻转为红方走子，将黑方的经验作为红方。
			LOGGER.info(f'black_state={black_state}, with action={black_action}, pi={np.max(black_pi):.3f}, to state={black_next_state}')

			# 这里得到黑方的走子，就可以马上开始跑我方的模型
			red_state = hoho_game.state
			red_pi, red_action = hoho_mcts.take_simulation(hoho_agent, hoho_game)
			red_next_state, red_z, _ = hoho_game.step(red_action)
			hoho_round.add_red_step(red_state, red_pi.tolist())
			LOGGER.info(f'red_state={red_state}, with action={red_action}, pi={np.max(red_pi):.3f}, to state={red_next_state}')

			red_move = convert_my_action_to_webgame_move(red_action)
			LOGGER.info(f'get red move={red_move}')
		json_data = {'Black': list(black_move), 'Red': list(red_move)}
		json_ = json.dumps(json_data)

		LOGGER.info(f'data size: replay buffer = {hoho_replay_buffer.size()} / round = {hoho_round.size()}')
		LOGGER.info(f'{round_count} rounds / {match_count} matches | elapse: {(time.time() - start_time):.3f}s')

		win_rate = (win_count / match_count) if match_count > 0 else 0
		LOGGER.info(f'model version: {hoho_agent.version} | win count: {win_count} | win rate: {win_rate}')
		LOGGER.info('========================================================')

	if not message_queue.empty():
		msg_info = message_queue.get()
		LOGGER.info(f'thread message: {msg_info}')
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
	# 	update_agent()
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


def update_agent():
	LOGGER.info('Start training!')

	# 模型训练
	agent_new, agent_current = train(hoho_agent)

	# 自博弈
	play_proc = mp.Process(target=self_battle, args=(agent_current, agent_new, True, message_queue))
	play_proc.start()


	# train_thread = threading.Thread(target=train, args=(hoho_agent, message_queue), name='hoho_train_thread')
	# train_thread.start()
	## train_thread.join()

	## mp.set_start_method('spawn')
	# train_proc = mp.Process(target=train, args=(hoho_agent, message_queue))
	# train_proc.start()
	## train_proc.join()


def find_top_version_model_path():
	root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	model_dir_path = os.path.join(root_dir_path, 'output', 'models')
	if not os.path.exists(model_dir_path):
		return None
		
	result_path = None
	top_version = 0
	for filename in os.listdir(model_dir_path):
		name = filename.split('.')[0]
		if not name.startswith(MODEL_FILE_PREFIX):
			continue

		items = name.split('_')
		if len(items) != 3:
			continue

		if int(items[2]) > top_version:
			top_version = int(items[2])
			result_path = os.path.join(model_dir_path, filename)
          
	return result_path
	

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
	setup_seed(2021)

	win_count = 0
	match_count = 0
	last_update_finish_time = 0
	# message_queue = queue.Queue()
	message_queue = mp.Queue()
	agent_updating = False
	agent_update_accepted = False
	agent_update_path = None
	hoho_mcts = None
	hoho_game = None
	hoho_round = None
	hoho_replay_buffer = ReplayBuffer()

	hoho_agent = Player()
	model_path = find_top_version_model_path()
	if model_path is not None:
		hoho_agent.load_model_from_path(model_path)

	LOGGER.info(f'[pid={os.getpid()}] start server!')
	start_server_(8000, 100)

	#hoho_test
	# update_agent()



	