
import json, os
import time
import queue

import torch
import torch.multiprocessing as mp
import numpy as np

from web.py_lib import chess, auto_chess
from .lib.http_ import Http_

from model.hoho_utils import *
from model.hoho_agent import *
from model.hoho_dqn import *
from model.hoho_mcts import *
from model.hoho_cchessgame import *
from model.hoho_config import *


def go_to_new_round(argv):
	global hoho_game, hoho_mcts, hoho_agent, hoho_round
	global match_count, agent_updating, agent_update_accepted, agent_update_path, win_count

	match_count += 1
	win_player = argv[2]

	if hoho_round is not None:
		if win_player == 'Red':
			win_count += 1
			hoho_round.update_winner('Red')
		elif win_player == 'Black':
			hoho_round.update_winner('Black')
		else: # 平局
			hoho_round.update_winner()

		# 必须update_winner之后再add_round!!!!!!
		if hoho_round.size() > 0:
			hoho_replay_buffer.add_round(hoho_round)

	hoho_round = Round(int(time.time()))

	if hoho_replay_buffer.round_size() >= 100:
		hoho_replay_buffer.save({'model_version': hoho_agent.version})
		hoho_replay_buffer.clear()

	if agent_update_accepted and (agent_update_path is not None):
		# hoho_agent = Player()

		# 为了好区分模型版本，模型更新前都先保存样本数据
		hoho_replay_buffer.save({'model_version': hoho_agent.version})
		hoho_replay_buffer.clear()
	
		agent_net_updated_count = hoho_agent.count
		
		hoho_agent = DQN(ACTION_DIM, LEARNING_RATE, GAMMA, EPSILON_G, TARGET_UPDATE_COUNT, DEVICE)
		hoho_agent.load_model_from_path(agent_update_path)
		hoho_agent.count = agent_net_updated_count
		LOGGER.info(f'Agent updated! version={hoho_agent.version}')
		
		agent_updating = False
		agent_update_accepted = False
		agent_update_path = None

	# hoho_mcts = MCTS(start_player=PLAYER_RED)
	hoho_game = CChessGame()

	red_state = hoho_game.state
	# pi, action = hoho_mcts.take_simulation(hoho_agent, hoho_game)
	red_action, red_pi = hoho_agent.take_action(red_state)
	mid_state, z, done = hoho_game.step(red_action)
	hoho_round.add_red_step(red_state, red_pi.tolist(), red_action, mid_state, z, done)

	move = convert_my_action_to_webgame_move(red_action)
	LOGGER.info(f'get red move={move}')

	return move

def go_on_gaming(func_name, data_board):
	global hoho_game, hoho_mcts, hoho_agent, hoho_round
	global match_count, agent_updating, agent_update_accepted, agent_update_path, win_count

	black_move = rpc_registry[func_name](*data_board)
	LOGGER.info(f'get black move={black_move}')  # 注意这里黑方走法，已经翻转了棋盘

	if black_move is None:
		black_move = ()
		LOGGER.error('black_move is None!')
	else:
		black_state = hoho_game.state
		black_action = convert_webgame_opponent_move_to_action(black_move)
	
		# 模型只关心红方，这里强制造一个确定性黑方走子策略
		black_pi = np.zeros((ACTION_DIM,))
		black_pi[ACTIONS_2_INDEX[black_action]] = 1.0

		# hoho_mcts.update_root_with_action(black_action)  # 独自更新MCTS的根节点，因为webgame选的black_action跟自己模型选的不一定一样
		black_mid_state, black_z, black_done = hoho_game.step(black_action)
		hoho_round.add_black_step(flip_board(black_state), flip_action_probas(black_pi).tolist(), flip_action(black_action), flip_board(black_mid_state), black_z, black_done)  # 注意：这里要翻转为红方走子，将黑方的经验作为红方。
		LOGGER.info(f"(flipped, base on red) black_state={flip_board(black_state)}, to state={flip_board(black_mid_state)}, with action={flip_action(black_action)}(not flip: {black_action})")

		if black_done:  # 黑方赢了，红方就不需要再走了
			LOGGER.info(f'black win!')
			return black_move, ()

		# 这里得到黑方的走子，就可以马上开始跑我方（红方）的模型
		red_state = hoho_game.state
		# red_pi, red_action = hoho_mcts.take_simulation(hoho_agent, hoho_game)
		red_action, red_pi = hoho_agent.take_action(red_state)
		red_mid_state, red_z, red_done = hoho_game.step(red_action)
		hoho_round.add_red_step(red_state, red_pi.tolist(), red_action, red_mid_state, red_z, red_done)
		LOGGER.info(f'red_state={red_state}, with action={red_action}, pi={np.max(red_pi):.3f}, to state={red_mid_state}')

		red_move = convert_my_action_to_webgame_move(red_action)
		LOGGER.info(f'get red move={red_move}')

		if red_done:
			LOGGER.info(f'red win!')

	return black_move, red_move


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
	global rpc_registry, agent_updating
	global hoho_agent, hoho_replay_buffer, hoho_round, updated_time
	params_ = request_.params_
	assert 'data' in params_, '服务请求参数中缺少 data'

	data = json.loads(params_['data'])
	assert 'func_name' in data, "'func_name' should be included in data"
	assert 'argv' in data, "'argv' should be included in data"

	func_name = data['func_name']
	assert func_name in rpc_registry, f'服务中没有登记函数 {func_name}, 所有函数: {", ".join(rpc_registry.keys())}'

	argv = data['argv']
	data_board = argv[0]
	round_count = argv[1]
	json_ = None

	if data_board == 'Action!': # 开始！
		red_move = go_to_new_round(argv)
		json_data = {'Red': list(red_move), 'expand:':{'agent_updating': agent_updating}}
		json_ = json.dumps(json_data)
	else:
		start_time = time.time()
		black_move, red_move = go_on_gaming(func_name, [data_board])   # data_board需要重新包一下
		json_data = {'Black': list(black_move), 'Red': list(red_move), 'expand': {'agent_updating': agent_updating}}
		json_ = json.dumps(json_data)

		LOGGER.info(f'current round steps = {hoho_round.size()} | total steps: {hoho_replay_buffer.step_size()} | total rounds: {hoho_replay_buffer.round_size()}')
		LOGGER.info(f'{round_count} rounds / {match_count} matches | elapse: {(time.time() - start_time):.3f}s')

		win_rate = (win_count / match_count) if match_count > 0 else 0
		LOGGER.info(f'model version: {hoho_agent.version} | win count: {win_count} | win rate: {win_rate}')
		LOGGER.info('========================================================')


	if (not agent_updating) and should_update_agent(hoho_agent.version):
		update_agent()
		updated_time = time.time()
		agent_updating = True

	# if not message_queue.empty():
	# 	msg_info = message_queue.get()
	# 	LOGGER.info(f'thread message: {msg_info}')
	# 	if msg_info.get(KEY_MSG_ID) == AGENT_MSG_ID_TRAIN_FINISH:
	# 		LOGGER.info(f'Agent training finish!')
	# 	elif msg_info.get(KEY_MSG_ID) == AGENT_MSG_ID_SELF_BATTLE_FINISH:
	# 		agent_update_accepted = msg_info.get(KEY_AGENT_ACCEPT)
	# 		agent_update_path = msg_info.get(KEY_MODEL_PATH)  # 等待开启新一局时再去update agent
			
	# 		LOGGER.info(f'Agent self-battle finish! update accepted: {agent_update_accepted}, model updated path: {agent_update_path}')

	return response_.write_response_JSON_OK_(json_)


def start_server_(port_, max_threads_):
	http_ = Http_(ip_='0.0.0.0', port_=port_, web_path_='web', max_threads_=max_threads_)
	http_.add_route_('/ajax', ajax_, 'GET')
	http_.add_route_('/ajax', ajax_, 'POST')
	http_.add_route_('/__dir__/{dir}', __dir__, 'GET')
	http_.add_route_('/__dir__/{dir}', __dir__, 'POST')
	http_.add_route_('/', home, 'GET')
	http_.start_()


def should_update_agent(model_version):
	global hoho_replay_buffer, updated_time

	# if hoho_replay_buffer.step_size() % 100:
	# 	return True

	if hoho_replay_buffer.step_size() > 100:     
		if time.time() - updated_time > 600:   # 大于500秒
			return True
	
	return False

	# root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	# data_dir_path = os.path.join(root_dir_path, 'output', 'data')
	# if not os.path.exists(data_dir_path):
	# 	return False

	# train_dataset = ChessDataset.load_from_dir(data_dir_path, version=model_version)
	# if len(train_dataset) < 25000:   # 少于多少条数据就不update了
	# 	return False

	# return True 


def update_agent():
	global hoho_agent, message_queue, agent_update_accepted, agent_update_path
	LOGGER.info('Start training!')

	# 模型训练
	# agent_new, agent_current = train(hoho_agent)

	agent_update_path = train_off_policy_agent(hoho_agent, 60, hoho_replay_buffer, batch_size = BATCH_SIZE) 
	agent_update_accepted = True


	# # 自博弈
	# play_proc = mp.Process(target=self_battle, args=(agent_current, agent_new, True, message_queue))
	# play_proc.start()

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
		items = name.split('_')
		if len(items) == 3:
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


def rpc_auto_move(board_key):
	board = chess.board_from_key(board_key)
	return auto_chess.auto_move(board)


rpc_registry = {}
# 登记函数 RPC auto_move
rpc_registry['rpc_auto_move'] = rpc_auto_move




if __name__ == '__main__':
	setup_seed(6666)

	mp.set_start_method('spawn')   # Unix上跑要打开这句！要写在所有执行多线程代码之前！

	win_count = 0
	match_count = 0
	message_queue = queue.Queue()
	agent_updating = False
	agent_update_accepted = False
	agent_update_path = None
	hoho_mcts = None
	hoho_game = None
	hoho_round = None
	updated_time = 0
	hoho_replay_buffer = ReplayBuffer()

	# hoho_agent = Player()
	hoho_agent = DQN(ACTION_DIM, LEARNING_RATE, GAMMA, EPSILON_G, TARGET_UPDATE_COUNT, DEVICE)
	model_path = find_top_version_model_path()
	LOGGER.info(f"model_path: {model_path}")
	
	if model_path is not None:
		hoho_agent.load_model_from_path(model_path)

	LOGGER.info(f'[pid={os.getpid()}] start server!')
	start_server_(8000, 100)

	#hoho_test
	# update_agent()



	